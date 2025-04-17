"""
This module defines the OmniRe environment model, which is a neural renderer
for rendering images based on the simulation state. It uses a trained model to generate
novel views of the environment based on the camera position and rotation.
It includes methods for rendering images, preparing frame data, and computing camera matrices.
"""

from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from sim_types import State
from path_utils import local_import_context

with local_import_context("drivestudio"):
    from drivestudio.datasets.driving_dataset import DrivingDataset
    from drivestudio.utils.misc import import_str
    from drivestudio.models.trainers import BasicTrainer

class EnvironmentModel:
    """Base class for the environment model."""
    def __init__(self):
        pass

    def get_sensor_output(self, state):
        """Generate sensor output for the given simulation state."""
        raise NotImplementedError("This method should be overridden by subclasses.")

class OmniReSetup:
    """
    OmniReSetup class that initializes the environment model with a trained neural renderer.
    It loads the configuration and checkpoint files, sets up the device,
    and prepares the dataset for rendering.
    """
    def __init__(self, config_path: str, checkpoint_path: str):
        config_path = Path(config_path)
        raw_checkpoint_path = Path(checkpoint_path)
        checkpoint_path = raw_checkpoint_path / "config.yaml"

        print(f"Loading config from: {config_path}")
        print(f"Loading checkpoint from: {raw_checkpoint_path}")

        # Load configuration
        self.data_cfg = OmegaConf.load(config_path)
        self.train_cfg = OmegaConf.load(checkpoint_path)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        print("Loading dataset...")
        self.data_cfg.data.data_root = self.data_cfg.data.data_root
        self.dataset = DrivingDataset(data_cfg=self.data_cfg.data)

        # Setup trainer
        self.trainer = import_str(self.train_cfg.trainer.type)(
            **self.train_cfg.trainer,
            num_timesteps=self.dataset.num_img_timesteps,
            model_config=self.train_cfg.model,
            num_train_images=len(self.dataset.train_image_set) if self.dataset.train_image_set else 0,
            num_full_images=len(self.dataset.full_image_set) if self.dataset.full_image_set else 0,
            test_set_indices=self.dataset.test_timesteps if hasattr(self.dataset, 'test_timesteps') else None,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3),
            device=self.device
        )
        
        ckpt_path = raw_checkpoint_path / "checkpoint_final.pth"

        # Load checkpoint
        self.trainer.resume_from_checkpoint(
            ckpt_path=ckpt_path,
            load_only_model=True
        )

        # Set to evaluation mode
        self.trainer.set_eval()

        # Cache for storing camera matrices
        self.camera_matrix_cache = {}

        print(f"OmniRe environment model initialized with checkpoint: {checkpoint_path}")

class OmniRe(EnvironmentModel):
    """
    OmniRe environment model class that initializes a neural renderer for rendering images
    based on the simulation state. It uses a trained model to generate novel views
    of the environment based on the camera position and rotation.
    It includes methods for rendering images, preparing frame data, and computing camera matrices.
    """
    def __init__(self, setup: OmniReSetup):
        """
        Initialize the OmniRe environment model with a trained neural renderer.
        
        Args:
            config_path (str): Path to the configuration YAML file
            checkpoint_path (str): Path to the model checkpoint
        """
        super(OmniRe, self).__init__()

        self.data_cfg = setup.data_cfg
        self.train_cfg = setup.train_cfg
        self.trainer = setup.trainer
        self.dataset = setup.dataset
        self.device = setup.device
        self.camera_matrix_cache = setup.camera_matrix_cache

    def render_single_frame(self, frame_data: dict) -> np.ndarray:
        """
        Render a single frame based on provided frame data.
        
        Args:
            frame_data (dict): Dictionary containing camera and image info for the frame
            
        Returns:
            np.ndarray: The rendered RGB image as a numpy array
        """
        with torch.no_grad():
            # Move data to GPU
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                if isinstance(value, torch.Tensor):
                    frame_data["image_infos"][key] = value.cuda(non_blocking=True)

            # Perform rendering
            outputs = self.trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True
            )

            # Extract RGB image and return
            rgb = outputs["rgb"].cpu().numpy().clip(
                min=1.e-6, max=1-1.e-6
            )

            # If depth is needed, you can extract it too
            if "depth" in outputs:
                depth = outputs["depth"].cpu().numpy()
                return rgb, depth

            return rgb

    def get_sensor_output(self, state):
        """
        Generate sensor output (RGB image) for the given simulation state.
        
        Args:
            state (dict): Current state of the simulation containing:
                - camera_position (np.ndarray): 3D position of the camera
                - camera_rotation (np.ndarray): Rotation of the camera (e.g., quaternion)
                - vehicle_positions (dict): Dictionary mapping vehicle IDs to positions
                - vehicle_rotations (dict): Dictionary mapping vehicle IDs to rotations
                - timestamp (float): Current simulation time
                
        Returns:
            dict: Sensor outputs including rendered image
        """
        # Prepare frame data for rendering based on current state
        frame_data = self.prepare_frame_data(state)

        # Render the image
        rgb_image = self.render_single_frame(frame_data)

        # Create sensor output dictionary
        sensor_output = {
            "rgb_image": rgb_image,
            # Add other sensor outputs as needed
        }

        return sensor_output

    def prepare_frame_data(self, state: State):
        """
        Prepare the frame data needed for rendering based on simulation state.
        
        Args:
            state (dict): Current state of the simulation
            
        Returns:
            dict: Frame data dictionary with cam_infos and image_infos
        """
        # Extract camera information
        camera_position = torch.tensor([
            state.ego_pos.x,
            state.ego_pos.y,
            state.ego_pos.z
        ], dtype=torch.float32)
        
        # Assuming state.ego_pos.heading is a scalar representing the yaw angle
        heading = torch.tensor(state.ego_pos.heading)
        half_yaw = heading / 2
        camera_rotation = torch.tensor([
            0.0,                     # x (roll)
            0.0,                     # y (pitch)
            torch.sin(half_yaw),    # z
            torch.cos(half_yaw)     # w
        ])
        
        timestamp = state.timestamp.time_us

        # Get camera matrix
        c2w = self.compute_camera_matrix(camera_position, camera_rotation)

        # Create camera information dictionary
        cam_infos = {
            "cam_id": torch.tensor([0], device=self.device),
            "cam_name": "sim_camera",
            "c2w": c2w,
            "width": torch.tensor([1280], device=self.device),
            "height": torch.tensor([720], device=self.device),
            "fov": torch.tensor([60.0], device=self.device),  # Field of view in degrees
            "timestamp": torch.tensor([timestamp], device=self.device),
            # Add other camera parameters required by your model
        }

        # Vehicle information
        vehicles = getattr(state, "vehicle_pos_list", {})
        
        # vehicles is on the form Position(x=664432.4584765462, y=3998282.198022293, z=0, heading=-1.5377766955975682
        
        vehicle_positions = {
            vehicle_id: np.array([vehicle.x, vehicle.y, vehicle.z], dtype=np.float32)
            for vehicle_id, vehicle in enumerate(vehicles)
        }
        vehicle_rotations = {
            vehicle_id: np.array([0.0, 0.0, np.sin(vehicle.heading / 2), np.cos(vehicle.heading / 2)], dtype=np.float32)
            for vehicle_id, vehicle in enumerate(vehicles)
        }

        # Create vehicle information for the renderer
        # This depends on how your renderer expects vehicle data
        # vehicle_data = {
        #     vehicle_id: {
        #         "position": torch.tensor(position, device=self.device),
        #         "rotation": torch.tensor(vehicle_rotations.get(vehicle_id, np.array([1.0, 0.0, 0.0, 0.0])), device=self.device),
        #     }
        #     for vehicle_id, position in vehicle_positions.items()
        # }
        
        vehicle_data = {
            vehicle_id: {
                "position": torch.tensor(position, device=self.device),
                "rotation": torch.tensor(vehicle_rotations.get(vehicle_id, np.array([1.0, 0.0, 0.0, 0.0])), device=self.device),
            }
            for vehicle_id, position in vehicle_positions.items()
        }

        # Create image information dictionary
        # This includes any additional data needed for rendering
        image_infos = {
            "timestamp": torch.tensor([timestamp], device=self.device),
            "vehicles": vehicle_data,
            # Add other required image information
        }

        return {
            "cam_infos": cam_infos,
            "image_infos": image_infos
        }

    def compute_camera_matrix(self, position, rotation):
        """
        Compute the camera-to-world transformation matrix from position and rotation.
        
        Args:
            position (np.ndarray): 3D position vector [x, y, z]
            rotation (np.ndarray): Rotation as quaternion [w, x, y, z]
            
        Returns:
            torch.Tensor: 4x4 camera-to-world transformation matrix
        """
        # Create a cache key for this position and rotation
        cache_key = (tuple(position), tuple(rotation))

        # Check if we've already computed this matrix
        if cache_key in self.camera_matrix_cache:
            return self.camera_matrix_cache[cache_key]

        # Convert position to tensor
        position_tensor = torch.tensor(position, dtype=torch.float32, device=self.device)

        # Initialize transformation matrix
        c2w = torch.eye(4, device=self.device)

        # Set translation component
        c2w[:3, 3] = position_tensor

        # Convert quaternion to rotation matrix
        # Assuming quaternion is [w, x, y, z]
        w, x, y, z = rotation

        # Construct rotation matrix from quaternion
        rot_matrix = torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], dtype=torch.float32, device=self.device)

        # Set rotation component of the transformation matrix
        c2w[:3, :3] = rot_matrix

        # Store in cache for future use
        self.camera_matrix_cache[cache_key] = c2w

        return c2w

    def update_vehicle_positions(self, state, vehicle_id, new_position, new_rotation):
        """
        Update the position and rotation of a specific vehicle in the state.
        
        Args:
            state (dict): Current simulation state
            vehicle_id (int): ID of the vehicle to update
            new_position (np.ndarray): New 3D position
            new_rotation (np.ndarray): New rotation (quaternion)
            
        Returns:
            dict: Updated state dictionary
        """
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()

        # Initialize vehicle dictionaries if they don't exist
        if "vehicle_positions" not in updated_state:
            updated_state["vehicle_positions"] = {}
        if "vehicle_rotations" not in updated_state:
            updated_state["vehicle_rotations"] = {}

        # Update vehicle position and rotation
        updated_state["vehicle_positions"][vehicle_id] = new_position
        updated_state["vehicle_rotations"][vehicle_id] = new_rotation

        return updated_state

    def create_trajectory(self, start_position, end_position, num_steps):
        """
        Create a linear trajectory between two positions.
        
        Args:
            start_position (np.ndarray): Starting position [x, y, z]
            end_position (np.ndarray): Ending position [x, y, z]
            num_steps (int): Number of steps in the trajectory
            
        Returns:
            np.ndarray: Array of positions along the trajectory
        """
        return np.linspace(start_position, end_position, num_steps)
