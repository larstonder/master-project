"""Python script to render images using the OmniRe model.
This script is designed to work with the DriveStudio framework and
is tailored for the NuPlan dataset.
It handles the rendering of images based on the simulation state
and the camera parameters.
"""

import os
import torch
import numpy as np
import torch
import cv2
import copy
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict
from scipy.spatial.transform import Rotation as R
from ..types.state_types import SystemState, EnvState, VehicleState
from ..utils.path_utils import local_import_context
from ..utils.misc import import_str  # pylint: disable=import-error
from .environment import Environment

with local_import_context("../master-project/drivestudio", True):
    # print current working directory    
    from datasets.driving_dataset import DrivingDataset  # py
    from datasets.base.pixel_source import get_rays
    from models.trainers.scene_graph import MultiTrainer

OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

class OmniReSetup:
    """
    OmniReSetup class that initializes the environment model with a trained neural renderer.
    It loads the configuration and checkpoint files, sets up the device,
    and prepares the dataset for rendering.
    """

    def __init__(
        self,
        data_path: str,
        checkpoint_path: str,
        cam_id: int = 0,
        start_timestep: int = 0,
    ):
        self.data_path = Path(data_path)
        print(f"Loading data from: {data_path}")

        self.cam_id = cam_id
        self.start_timestep = start_timestep

        self.cam_to_ego = self._load_and_transform_extrinsics()
        self.world_to_ego = self._calculate_world_anchor_transform()

        print(f"Data loaded.")

        raw_checkpoint_path = Path(checkpoint_path)
        checkpoint_path = raw_checkpoint_path / "config.yaml"

        print(f"Loading checkpoint from: {raw_checkpoint_path}")

        # Load configuration
        # self.data_cfg = OmegaConf.load(config_path)
        self.cfg = OmegaConf.load(checkpoint_path)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = DrivingDataset(data_cfg=self.cfg.data)

        print(f"Initializing trainer {self.cfg.trainer.type}")

        # Setup trainer
        self.trainer = MultiTrainer(
            **self.cfg.trainer,
            num_timesteps=self.dataset.num_img_timesteps,
            model_config=self.cfg.model,
            num_train_images=len(self.dataset.train_image_set),
            num_full_images=len(self.dataset.full_image_set),
            test_set_indices=self.dataset.test_timesteps,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3),
            device=self.device,
        )

        # Evaluate that the trainer is initialized correctly
        if self.trainer is None:
            raise ValueError("Trainer is not initialized correctly")

        print(f"Trainer type: {type(self.trainer)}")

        print("TRAINER INIT DONE")

        ckpt_path = raw_checkpoint_path / "checkpoint_final.pth"

        # Load checkpoint
        self.trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)

        print(
            f"Resuming from checkpoint: {ckpt_path}, starting at step {self.trainer.step}"
        )

    def _load_and_transform_extrinsics(self) -> np.ndarray:
        """Loads camera extrinsics and applies OPENCV2DATASET transformation."""
        raw_cam_to_ego_path = os.path.join(
            self.data_path, "extrinsics", f"{self.cam_id}.txt"
        )

        if not os.path.exists(raw_cam_to_ego_path):
            raise FileNotFoundError(f"Extrinsics file not found: {raw_cam_to_ego_path}")

        raw_extrinsics = np.loadtxt(raw_cam_to_ego_path)
        cam_to_ego = raw_extrinsics @ OPENCV2DATASET
        return cam_to_ego

    def _calculate_world_anchor_transform(self) -> np.ndarray:
        """Calculates the transformation from nuPlan sim world to OmniRe world."""
        ego_pose_start_path = os.path.join(
            self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt"
        )
        if not os.path.exists(ego_pose_start_path):
            raise FileNotFoundError(
                f"OmniRe reference ego_pose file not found: {ego_pose_start_path}"
            )
        ego_to_world_start = np.loadtxt(ego_pose_start_path)
        world_to_ego = np.linalg.inv(ego_to_world_start)
        return world_to_ego

class OmniRe(Environment):
    """
    OmniReModel class for rendering images using the OmniRe model.
    This class is designed to work with the DriveStudio framework and
    is tailored for the NuPlan dataset.
    It handles the rendering of images based on the simulation state
    and the camera parameters.
    """

    def __init__(self, setup: OmniReSetup, original_state: SystemState):
        super().__init__()
        self.cfg = setup.cfg
        self.device = setup.device
        self.trainer: MultiTrainer = setup.trainer
        self.dataset = setup.dataset
        self._initialize()

    def _initialize(self):
        """Initialize the dataset and frame data"""
        render_traj = self.dataset.get_novel_render_traj()
        self.frame_data = self.dataset.pixel_source.prepare_novel_view_render_data(
            self.dataset.pixel_source, render_traj["front_center_interp"]
        )[0]

    def _get_delta_pos(
        self, last_pos: VehicleState, current_pos: VehicleState
    ) -> VehicleState:
        """Get the delta position between two positions"""
        delta_x = float(current_pos.x) - float(last_pos.x)
        delta_y = float(current_pos.y) - float(last_pos.y)
        delta_z = float(current_pos.z) - float(last_pos.z)
        delta_heading = float(current_pos.heading) - float(last_pos.heading)
        return VehicleState(
            x=delta_x, y=delta_y, z=delta_z, heading=delta_heading, id=current_pos.id
        )

    def _edit_nodes(self, vehicle_deltas: list[VehicleState] = None):
        """Edit the nodes in the scene graph based on vehicle position deltas.

        Args:
            vehicle_deltas: List of VehicleState objects representing the change in position and heading for each vehicle.
        """
        try:
            if vehicle_deltas is None or len(vehicle_deltas) == 0:
                print("No vehicle deltas provided, skipping node editing")
                return

            print(f"=== Applying changes to {len(vehicle_deltas)} vehicles ===")

            if "RigidNodes" in self.trainer.gaussian_classes.keys():
                rigid_model = self.trainer.models["RigidNodes"]

                with torch.no_grad():  # Disable gradient tracking during editing
                    for delta in vehicle_deltas:
                        instance_id = delta.id

                        if not hasattr(rigid_model, "dataset_id_to_model_id_map"):
                            print(
                                "Error: RigidNodes instance does not have 'dataset_id_to_model_id_map'. Skipping edit for this vehicle."
                            )
                            return

                        if instance_id in rigid_model.dataset_id_to_model_id_map:
                            model_ins_id = rigid_model.dataset_id_to_model_id_map[
                                instance_id
                            ]

                            x, y, z, heading = delta.x, delta.y, delta.z, delta.heading

                            # Apply translation if x, y, z has changed
                            if x != 0 or y != 0 or z != 0:
                                translation = torch.tensor(
                                    [x, y, z], device=self.device
                                )
                                print(
                                    f"Translating model instance {model_ins_id} (from dataset id {instance_id}) by {translation}"
                                )
                                rigid_model.translate_instance(
                                    model_ins_id, translation
                                )

                            # Apply rotation if heading has changed
                            if heading != 0:
                                # Convert heading change (in radians) to quaternion rotation around Z-axis
                                delta_heading_rot = R.from_euler(
                                    "y", heading, degrees=False
                                )
                                rotation_quat_np = delta_heading_rot.as_quat()
                                # Convert NumPy array to PyTorch tensor before calling rotate_instance
                                rotation_quat = torch.tensor(
                                    rotation_quat_np,
                                    device=self.device,
                                    dtype=torch.float32,
                                )
                                print(
                                    f"Rotating model instance {model_ins_id} (from dataset id {instance_id}) by {rotation_quat}"
                                )
                                rigid_model.rotate_instance(model_ins_id, rotation_quat)
                        else:
                            print(
                                f"Dataset instance ID {instance_id} not found in RigidNodes map. Available dataset IDs in map: {list(rigid_model.dataset_id_to_model_id_map.keys())}. Skipping edit for this vehicle."
                            )
                            continue

                print("Vehicle transformations completed")
            else:
                print("No RigidNodes found in scene graph")

        except Exception as e:
            print(f"Error in edit_nodes: {str(e)}")
            traceback.print_exc()

    def get_sensor_output(
        self, original_state: SystemState, last_state: SystemState, current_state: SystemState
    ) -> EnvState:
        frame_data = copy.deepcopy(self.frame_data)

        # ---- UPDATE OTHER VEHICLES ----
        vehicle_deltas = []

        # Safely compare vehicle lists to avoid index errors
        if hasattr(current_state, "vehicle_pos_list") and hasattr(
            last_state, "vehicle_pos_list"
        ):
            # Get the common length to avoid index out of range
            for i in range(len(current_state.vehicle_pos_list)):

                current_id_to_check = current_state.vehicle_pos_list[i].id
                for j in range(len(last_state.vehicle_pos_list)):
                    if last_state.vehicle_pos_list[j].id == current_id_to_check:
                        delta_pos = self._get_delta_pos(
                            last_state.vehicle_pos_list[j],
                            current_state.vehicle_pos_list[i],
                        )
                        vehicle_deltas.append(delta_pos)
                        break

        self._edit_nodes(vehicle_deltas)

        # ---- UPDATE CAMERA POSITION ----

        ego_delta = self._get_delta_pos(original_state.ego_pos, current_state.ego_pos)

        c2w = frame_data["cam_infos"]["camera_to_world"].to(self.device)

        c2w[:3, 3] = c2w[:3, 3] + torch.tensor(
            [
                float(-ego_delta.y),
                float(-ego_delta.x),
                float(ego_delta.z),
            ],  # Example of axis reordering
            device=self.device,
        )

        current_rotation = c2w[:3, :3].cpu().numpy()

        current_rot_obj = R.from_matrix(current_rotation)
        delta_heading_rot = R.from_euler("y", ego_delta.heading, degrees=False)

        new_rotation = current_rot_obj * delta_heading_rot
        c2w[:3, :3] = torch.tensor(new_rotation.as_matrix(), device=self.device)

        frame_data["cam_infos"]["camera_to_world"] = c2w

        cam_id = 0

        H, W = (
            self.dataset.pixel_source.camera_data[cam_id].HEIGHT,
            self.dataset.pixel_source.camera_data[cam_id].WIDTH,
        )

        x, y = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="xy",
        )
        x, y = x.flatten(), y.flatten()

        intrinsics = frame_data["cam_infos"]["intrinsics"]

        x, y, c2w, intrinsics = (
            x.to(self.device),
            y.to(self.device),
            c2w.to(self.device),
            intrinsics.to(self.device),
        )

        origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsics)
        origins = origins.reshape(H, W, 3)
        viewdirs = viewdirs.reshape(H, W, 3)
        direction_norm = direction_norm.reshape(H, W, 1)

        frame_data["image_infos"]["origins"] = origins
        frame_data["image_infos"]["viewdirs"] = viewdirs
        frame_data["image_infos"]["direction_norm"] = direction_norm

        self.trainer.set_eval()

        with torch.no_grad():
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][key] = value.cuda(non_blocking=True)

            outputs = self.trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True,
            )

            rgb = outputs["rgb"].cpu().numpy().clip(min=1.0e-6, max=1 - 1.0e-6)

            return EnvState(rgb_image=rgb, depth=None)
