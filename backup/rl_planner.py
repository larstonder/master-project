import math
import torch
import torch.nn as nn
from typing import Type, Optional, Union
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from .abstract_planner import AbstractPlanner
from ..types.state_types import EnvState, VehicleState, SystemState
from ..types.action import Action, Trajectory, Waypoint
from ..types.observation_type import Observation


class RGBObservation(Observation):
    """
    Observation type for RGB images used by the reinforcement learning planner.
    """
    pass


class ReinforcedPlanner(AbstractPlanner):
    """
    Planner implementing a reinforcement learning agent.
    Uses a CNN to process RGB images and outputs steering commands and acceleration.
    The output is converted to a trajectory with waypoints using realistic vehicle dynamics.
    
    :param hidden_dimensions: defines the number of hidden dimensions and the channels in each CNN layer.
    :param image_dimension: [height, width] Defines the dimensions of the input image
    :param horizon_seconds: Time horizon for planning
    :param sampling_time: Time step for trajectory sampling
    :param base_velocity: Base velocity for the vehicle when acceleration is 0
    :param max_steering_angle: Maximum steering angle in radians
    :param max_acceleration: Maximum acceleration in m/s²
    :param wheelbase: Vehicle wheelbase for bicycle model (meters)
    :param linear_width: width of the linear layer before output
    :param batch_size: Batch size for training
    :param num_epochs: Number of epochs for training
    :param lr: learning rate
    :param criterion: Loss function used to train the model
    :param optimizer_type: Type of optimizer to use (Adam, SGD, etc.)
    NOTE: Consider using imitation learning algorithms like DAgger or behavioral cloning
    """
    
    def __init__(
        self,
        hidden_dimensions: list[int], 
        horizon_seconds: float = 5.0,
        sampling_time: float = 0.1,
        base_velocity: float = 5.0,
        max_steering_angle: float = math.pi / 6,  # 30 degrees
        max_acceleration: float = 3.0,  # m/s²
        wheelbase: float = 2.7,  # meters, typical car wheelbase
        image_dimension: tuple[int, int] = (360, 640), 
        linear_width: int = 512, 
        batch_size: int = 1,
        num_epochs: int = 1,
        lr: float = 1e-3,
        criterion: _Loss = nn.MSELoss(),
        optimizer_type: str = "Adam",
    ):
        """
        Constructor for ReinforcedPlanner.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 3  # RGB image
        self.output_dimension = 2  # [steering_angle, acceleration]
        self.hidden_dimensions = hidden_dimensions
        self.image_dimension = image_dimension
        self.linear_width = linear_width
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        
        # Vehicle and trajectory parameters
        self.horizon_seconds = horizon_seconds
        self.sampling_time = sampling_time
        self.base_velocity = base_velocity
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration
        self.wheelbase = wheelbase
        
        # Initialize the neural network
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Initialize optimizer - will be created when needed
        self.optimizer = None
        self._create_optimizer()

    def _create_optimizer(self):
        """Create optimizer with model parameters."""
        if self.optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _build_model(self) -> nn.Module:
        """
        Build the neural network model for the RL agent.
        Uses a more efficient architecture suitable for real-time inference.
        """
        layers = []
        in_channels = self.input_channels
        
        # Efficient CNN backbone inspired by MobileNet
        for i, hidden_dim in enumerate(self.hidden_dimensions):
            if i == 0:
                # First layer with larger receptive field
                layers.extend([
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                ])
            else:
                # Depthwise separable convolutions for efficiency
                layers.extend([
                    # Depthwise convolution
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, 
                             groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    # Pointwise convolution
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                ])
            in_channels = hidden_dim
        
        # Global average pooling and final layers
        layers.extend([
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed size output
            nn.Flatten(),
            nn.Linear(self.hidden_dimensions[-1] * 16, self.linear_width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.linear_width, self.linear_width // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.linear_width // 2, self.output_dimension),
            nn.Tanh()  # Output values between -1 and 1
        ])
        
        return nn.Sequential(*layers)

    def initialize(self, initialization) -> None:
        """
        Initialize the planner with any required setup.
        """
        pass

    @property
    def name(self) -> str:
        """
        Return the name of the planner.
        """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """
        Return the type of observation that this planner expects.
        """
        return RGBObservation

    def compute_trajectory(self, current_input):
        """
        Computes the ego vehicle trajectory
        :param current_input: Current input containing the system state
        :return: Action containing the planned trajectory
        """
        return self.compute_planner_trajectory(current_input)

    def compute_planner_trajectory(self, current_input) -> Action:
        """
        Compute the trajectory based on the current environment state.
        :param current_input: The current environment state containing RGB image and depth information
        :return: An Action object containing a trajectory with waypoints
        """
        # Extract current state information with better error handling
        current_pos, timestamp, rgb_image, current_velocity = self._extract_state_info(current_input)
        
        if rgb_image is None:
            raise ValueError("RGB image is required for ReinforcedPlanner")
        
        # Process image and get NN predictions
        steering_angle, acceleration = self._predict_actions(rgb_image)
        
        # Generate trajectory using bicycle model
        waypoints = self._generate_trajectory_bicycle_model(
            current_pos, steering_angle, acceleration, current_velocity, timestamp
        )
        
        # Create trajectory and action
        trajectory = Trajectory(waypoints)
        action = Action(trajectory)
        
        return action

    def _extract_state_info(self, current_input) -> tuple[VehicleState, float, Optional[object], float]:
        """
        Extract state information from different input types.
        
        :param current_input: Input state (SystemState, EnvState, or other)
        :return: Tuple of (current_pos, timestamp, rgb_image, current_velocity)
        """
        if hasattr(current_input, 'ego_pos'):
            # SystemState input
            current_pos = current_input.ego_pos
            timestamp = current_input.timestamp
            rgb_image = getattr(current_input, 'rgb_image', None)
            # Try to get velocity from ego_pos, fallback to base_velocity
            current_velocity = getattr(current_pos, 'velocity', self.base_velocity)
        elif hasattr(current_input, 'rgb_image'):
            # EnvState input
            rgb_image = current_input.rgb_image
            timestamp = current_input.timestamp
            current_pos = VehicleState(0.0, 0.0, 0.0, 0.0, 0)
            current_velocity = self.base_velocity
        elif hasattr(current_input, 'timestamp'):
            # Generic input with timestamp
            timestamp = current_input.timestamp
            rgb_image = getattr(current_input, 'rgb_image', None)
            current_pos = VehicleState(0.0, 0.0, 0.0, 0.0, 0)
            current_velocity = self.base_velocity
        else:
            # Fallback for unknown input types
            import time
            timestamp = time.time()
            rgb_image = getattr(current_input, 'rgb_image', None)
            current_pos = VehicleState(0.0, 0.0, 0.0, 0.0, 0)
            current_velocity = self.base_velocity
            
        return current_pos, timestamp, rgb_image, current_velocity

    def _predict_actions(self, rgb_image) -> tuple[float, float]:
        """
        Process RGB image and predict steering angle and acceleration.
        
        :param rgb_image: RGB image array
        :return: Tuple of (steering_angle, acceleration)
        """
        # Convert RGB image to tensor and normalize
        rgb_tensor = torch.from_numpy(rgb_image).float()
        
        # Handle different image formats
        if len(rgb_tensor.shape) == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)  # HWC to CHW
        elif len(rgb_tensor.shape) == 4:
            rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # NHWC to NCHW
        
        # Normalize to [0, 1] if values are in [0, 255]
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
            
        # Add batch dimension if needed
        if len(rgb_tensor.shape) == 3:
            rgb_tensor = rgb_tensor.unsqueeze(0)
            
        rgb_tensor = rgb_tensor.to(self.device)

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(rgb_tensor)
            steering_normalized = outputs[0, 0].item()  # Output is in [-1, 1]
            accel_normalized = outputs[0, 1].item()     # Output is in [-1, 1]
        
        # Scale outputs to vehicle limits
        steering_angle = steering_normalized * self.max_steering_angle
        acceleration = accel_normalized * self.max_acceleration
        
        return steering_angle, acceleration

    def _generate_trajectory_bicycle_model(
        self, 
        current_pos: VehicleState, 
        steering_angle: float, 
        acceleration: float, 
        current_velocity: float,
        timestamp: float
    ) -> list[Waypoint]:
        """
        Generate trajectory waypoints using bicycle model for more realistic vehicle dynamics.
        
        :param current_pos: Current vehicle position
        :param steering_angle: Steering angle command in radians
        :param acceleration: Acceleration command in m/s²
        :param current_velocity: Current vehicle velocity
        :param timestamp: Current timestamp
        :return: List of waypoints forming the trajectory
        """
        waypoints = []
        
        # Calculate number of steps
        num_steps = int(self.horizon_seconds / self.sampling_time)
        
        # Starting state
        x = current_pos.x if hasattr(current_pos, 'x') else 0.0
        y = current_pos.y if hasattr(current_pos, 'y') else 0.0
        heading = current_pos.heading if hasattr(current_pos, 'heading') else 0.0
        velocity = current_velocity
        
        for i in range(num_steps + 1):
            # Create waypoint at current position
            waypoint = Waypoint(x, y, heading)
            waypoints.append(waypoint)
            
            if i < num_steps:  # Don't update on the last iteration
                # Update velocity using acceleration
                velocity += acceleration * self.sampling_time
                velocity = max(0.1, min(velocity, 25.0))  # Clamp velocity to reasonable range
                
                # Bicycle model kinematics
                # Calculate turn radius based on steering angle and wheelbase
                if abs(steering_angle) > 1e-6:  # Avoid division by zero
                    turn_radius = self.wheelbase / math.tan(abs(steering_angle))
                    angular_velocity = velocity / turn_radius
                    if steering_angle < 0:
                        angular_velocity = -angular_velocity
                else:
                    angular_velocity = 0.0
                
                # Update heading
                heading += angular_velocity * self.sampling_time
                
                # Update position
                x += velocity * math.cos(heading) * self.sampling_time
                y += velocity * math.sin(heading) * self.sampling_time
        
        return waypoints

    def train_step(self, rgb_image: torch.Tensor, target_actions: torch.Tensor) -> float:
        """
        Perform a single training step.
        
        :param rgb_image: Input RGB image tensor [batch_size, 3, H, W]
        :param target_actions: Target actions [batch_size, 2] (steering, acceleration)
        :return: Loss value
        """
        if self.optimizer is None:
            self._create_optimizer()
            
        self.model.train()
        
        # Move tensors to device
        rgb_image = rgb_image.to(self.device)
        target_actions = target_actions.to(self.device)
        
        # Forward pass
        outputs = self.model(rgb_image)
        
        # Calculate loss
        loss = self.criterion(outputs, target_actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, rgb_image: torch.Tensor, target_actions: torch.Tensor) -> dict:
        """
        Evaluate the model on given data.
        
        :param rgb_image: Input RGB image tensor
        :param target_actions: Target actions tensor
        :return: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            rgb_image = rgb_image.to(self.device)
            target_actions = target_actions.to(self.device)
            
            outputs = self.model(rgb_image)
            loss = self.criterion(outputs, target_actions)
            
            # Calculate additional metrics
            mae = torch.mean(torch.abs(outputs - target_actions))
            
            # Per-output metrics
            steering_mae = torch.mean(torch.abs(outputs[:, 0] - target_actions[:, 0]))
            accel_mae = torch.mean(torch.abs(outputs[:, 1] - target_actions[:, 1]))
            
        return {
            'loss': loss.item(),
            'mae': mae.item(),
            'steering_mae': steering_mae.item(),
            'acceleration_mae': accel_mae.item()
        }

    def save_model(self, path: str):
        """
        Save the model state dict and configuration.
        
        :param path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'hidden_dimensions': self.hidden_dimensions,
                'image_dimension': self.image_dimension,
                'linear_width': self.linear_width,
                'horizon_seconds': self.horizon_seconds,
                'sampling_time': self.sampling_time,
                'base_velocity': self.base_velocity,
                'max_steering_angle': self.max_steering_angle,
                'max_acceleration': self.max_acceleration,
                'wheelbase': self.wheelbase,
                'lr': self.lr,
                'optimizer_type': self.optimizer_type,
            }
        }, path)

    def load_model(self, path: str):
        """
        Load the model state dict and configuration.
        
        :param path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load configuration if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    @classmethod
    def from_checkpoint(cls, path: str) -> 'ReinforcedPlanner':
        """
        Create a ReinforcedPlanner instance from a saved checkpoint.
        
        :param path: Path to the checkpoint
        :return: ReinforcedPlanner instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        
        # Create instance with loaded config
        planner = cls(**config)
        planner.load_model(path)
        
        return planner


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create planner with realistic parameters
    planner = ReinforcedPlanner(
        hidden_dimensions=[32, 64, 128],
        horizon_seconds=3.0,
        sampling_time=0.1,
        base_velocity=5.0,
        max_steering_angle=math.pi/6,  # 30 degrees
        max_acceleration=2.0,  # 2 m/s²
        wheelbase=2.7,  # meters
        image_dimension=(240, 320),
        linear_width=256,
        lr=1e-4,
        optimizer_type="Adam"
    )
    
    # Initialize planner
    planner.initialize(None)
    
    # Create a dummy RGB image
    rgb_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # Create environment state
    env_state = EnvState(
        rgb_image=rgb_image,
        depth=np.zeros((240, 320)),
        timestamp=0.0
    )
    
    # Generate trajectory
    action = planner.compute_planner_trajectory(env_state)
    
    print(f"Generated trajectory with {len(action.trajectory.waypoints)} waypoints")
    for i, waypoint in enumerate(action.trajectory.waypoints[:5]):  # Print first 5 waypoints
        print(f"Waypoint {i}: x={waypoint.x:.2f}, y={waypoint.y:.2f}, heading={waypoint.heading:.2f}")
    
    # Example training step
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 240, 320)
    dummy_targets = torch.randn(batch_size, 2)  # [steering, acceleration]
    
    loss = planner.train_step(dummy_images, dummy_targets)
    print(f"Training loss: {loss:.4f}")
    
    # Example evaluation
    metrics = planner.evaluate(dummy_images, dummy_targets)
    print(f"Evaluation metrics: {metrics}")