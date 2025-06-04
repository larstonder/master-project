import math
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Type, Optional, Union, List, Any
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from monarch.planning.abstract_planner import AbstractPlanner
from monarch.typings.state_types import EnvState, VehicleState, SystemState, VehicleParameters
from monarch.typings.trajectory import Trajectory, Waypoint


class ReinforcementLearningPlanner(AbstractPlanner):
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
        hidden_dimensions: List[int] = [32, 64, 128], 
        horizon_seconds: float = 5.0,
        sampling_time: float = 0.1,
        base_velocity: float = 5.0,
        max_steering_angle: float = math.pi / 6,  # 30 degrees
        max_acceleration: float = 3.0,  # m/s²
        image_dimension: tuple[int, int] = (360, 640), 
        linear_width: int = 512, 
        batch_size: int = 1,
        num_epochs: int = 1,
        lr: float = 1e-3,
        criterion: _Loss = nn.MSELoss(),
        optimizer_type: str = "Adam",
    ):
        """
        Constructor for ReinforcementLearningPlanner.
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
        self.wheelbase = get_pacifica_parameters().wheel_base
        
        # Initialize the neural network
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Initialize optimizer - will be created when needed
        self.optimizer = None
        self._create_optimizer()
        
        # Training history
        self.training_losses = []
        self.evaluation_metrics = []

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
        # Store initialization parameters if needed
        self._initialization = initialization
        print(f"Initialized {self.name} with device: {self.device}")

    @property
    def name(self) -> str:
        """
        Return the name of the planner.
        """
        return self.__class__.__name__

    def observation_type(self) -> Type:
        """Inherited, see superclass"""
        # Return a generic type since we don't use specific observation types like DetectionsTracks
        return EnvState

    def compute_planner_trajectory(self, env_state: EnvState, state_history: List[SystemState]) -> Trajectory:
        """
        Compute the trajectory based on the current environment state.
        :param env_state: The current environment state containing RGB image and depth information
        :param state_history: List of system states with ego position and vehicle parameters
        :return: A Trajectory object containing a trajectory with waypoints
        """
        # Extract current state information from state_history
        current_system_state = state_history[-1]
        current_pos = current_system_state.ego_pos
        timestamp = current_system_state.timestamp
        
        # Get RGB image from env_state
        rgb_image = getattr(env_state, 'rgb_image', None)
        
        # Get current velocity from vehicle parameters
        current_velocity = math.sqrt(
            current_pos.vehicle_parameters.vx**2 + current_pos.vehicle_parameters.vy**2
        ) if hasattr(current_pos, 'vehicle_parameters') else self.base_velocity
        
        if rgb_image is None:
            # If no RGB image is available, use a default policy (e.g., go straight)
            print("Warning: No RGB image available, using default straight trajectory")
            steering_angle = 0.0
            acceleration = 0.0
        else:
            # Process image and get NN predictions
            steering_angle, acceleration = self._predict_actions(rgb_image)
        
        # Generate trajectory using bicycle model
        waypoints = self._generate_trajectory_bicycle_model(
            current_pos, steering_angle, acceleration, current_velocity, timestamp
        )
        
        # Create trajectory
        trajectory = Trajectory(waypoints)
        
        return trajectory

    def _calculate_angular_velocity_from_steering(self, velocity: float, steering_angle: float) -> float:
        """
        Calculate angular velocity using bicycle model from velocity and steering angle.
        
        :param velocity: Current velocity in m/s
        :param steering_angle: Steering angle in radians
        :return: Angular velocity in rad/s
        """
        if abs(steering_angle) > 1e-6 and velocity > 0.1:  # Avoid division by zero
            turn_radius = self.wheelbase / math.tan(abs(steering_angle))
            angular_velocity = velocity / turn_radius
            if steering_angle < 0:
                angular_velocity = -angular_velocity
            return angular_velocity
        else:
            return 0.0

    def _predict_actions(self, rgb_image) -> tuple[float, float]:
        """
        Process RGB image and predict steering angle and acceleration.
        
        :param rgb_image: RGB image array
        :return: Tuple of (steering_angle, acceleration)
        """
        try:
            # Convert RGB image to tensor and normalize
            if isinstance(rgb_image, np.ndarray):
                rgb_tensor = torch.from_numpy(rgb_image).float()
            else:
                rgb_tensor = torch.tensor(rgb_image).float()
            
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
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return safe defaults
            return 0.0, 0.0

    def _generate_trajectory_bicycle_model(
        self, 
        current_pos: VehicleState, 
        steering_angle: float, 
        acceleration: float, 
        current_velocity: float,
        timestamp: float
    ) -> List[Waypoint]:
        """
        Generate trajectory waypoints using bicycle model for more realistic vehicle dynamics.
        Enhanced with proper angular velocity calculation and tracking.
        
        :param current_pos: Current vehicle position
        :param steering_angle: Steering angle command in radians
        :param acceleration: Acceleration command in m/s²
        :param current_velocity: Current vehicle velocity
        :param timestamp: Current timestamp in microseconds
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
        
        # Initialize angular velocity from current state or calculate from steering
        if hasattr(current_pos.vehicle_parameters, 'angular_velocity'):
            current_angular_velocity = current_pos.vehicle_parameters.angular_velocity
        else:
            current_angular_velocity = self._calculate_angular_velocity_from_steering(velocity, steering_angle)
        
        for i in range(num_steps + 1):
            # Calculate velocity components at current state
            vx = velocity * math.cos(heading)
            vy = velocity * math.sin(heading)
            current_timestamp = timestamp + i * self.sampling_time * 1e6
            
            # Create waypoint at current position with all required parameters
            # Note: Waypoint constructor expects (x, y, heading, vx, vy, angular_velocity, timestamp)
            waypoint = Waypoint(x, y, heading, vx, vy, current_angular_velocity, current_timestamp)
            waypoints.append(waypoint)
            
            if i < num_steps:  # Don't update on the last iteration
                # Update velocity using acceleration
                new_velocity = velocity + acceleration * self.sampling_time
                new_velocity = max(0.1, min(new_velocity, 25.0))  # Clamp velocity to reasonable range
                
                # Calculate angular velocity using bicycle model
                angular_velocity = self._calculate_angular_velocity_from_steering(new_velocity, steering_angle)
                
                # Smooth angular velocity changes to avoid jerky motion
                alpha = 0.7  # Smoothing factor
                angular_velocity = alpha * angular_velocity + (1 - alpha) * current_angular_velocity
                
                # Update heading using calculated angular velocity
                heading += angular_velocity * self.sampling_time
                
                # Normalize heading to [-pi, pi]
                while heading > math.pi:
                    heading -= 2 * math.pi
                while heading < -math.pi:
                    heading += 2 * math.pi
                
                # Update position using new velocity and heading
                x += new_velocity * math.cos(heading) * self.sampling_time
                y += new_velocity * math.sin(heading) * self.sampling_time
                
                # Update for next iteration
                velocity = new_velocity
                current_angular_velocity = angular_velocity
        
        return waypoints

    def train_step_from_evaluation(self, rgb_image: torch.Tensor, actions_taken: torch.Tensor, rewards: torch.Tensor) -> float:
        """
        Perform a single training step using policy gradient with evaluation-based rewards.
        
        :param rgb_image: Input RGB image tensor [batch_size, 3, H, W]
        :param actions_taken: Actions that were taken [batch_size, 2] (steering, acceleration)
        :param rewards: Evaluation scores/rewards for those actions [batch_size]
        :return: Loss value
        """
        if self.optimizer is None:
            self._create_optimizer()
            
        self.model.train()
        
        # Move tensors to device
        rgb_image = rgb_image.to(self.device)
        actions_taken = actions_taken.to(self.device)
        rewards = rewards.to(self.device)
        
        # Forward pass to get trajectory probabilities
        outputs = self.model(rgb_image)
        
        # Calculate log probabilities for the actions taken
        # Since we use tanh output, we need to handle the continuous action space
        # Use a Gaussian policy assumption with fixed variance
        action_variance = 0.1  # Fixed variance for simplicity
        
        # Calculate negative log likelihood (assuming Gaussian distribution)
        log_probs = -0.5 * torch.sum(((outputs - actions_taken) ** 2) / action_variance, dim=1)
        
        # Normalize rewards (subtract baseline to reduce variance)
        if len(rewards) > 1:
            reward_baseline = torch.mean(rewards)
            advantages = rewards - reward_baseline
        else:
            advantages = rewards
        
        # Policy gradient loss: -log_prob * advantage
        policy_loss = -torch.mean(log_probs * advantages)
        
        # Add entropy bonus to encourage exploration
        entropy_bonus = 0.01 * torch.mean(torch.sum(outputs ** 2, dim=1))
        
        total_loss = policy_loss + entropy_bonus
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        loss_value = total_loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value

    def train_episode_from_evaluation(self, trajectory_data: List[dict]) -> dict:
        """
        Train on an episode using evaluation-based rewards.
        
        :param trajectory_data: List of dicts with keys 'rgb_image', 'trajectory', 'reward'
        :return: Training statistics
        """
        if len(trajectory_data) == 0:
            return {'average_loss': 0.0, 'total_loss': 0.0, 'num_batches': 0}
        
        # Prepare data
        rgb_images = []
        trajectories = []
        rewards = []
        
        for data_point in trajectory_data:
            rgb_images.append(data_point['rgb_image'])
            trajectories.append(data_point['trajectory'])
            rewards.append(data_point['reward'])
        
        # Convert to tensors
        image_tensor = self._prepare_image_batch(rgb_images)
        trajectory_tensor = torch.tensor(trajectories, dtype=torch.float32)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.num_epochs):
            for i in range(0, len(trajectory_data), self.batch_size):
                # Prepare batch
                batch_images = image_tensor[i:i+self.batch_size]
                batch_trajectories = trajectory_tensor[i:i+self.batch_size]
                batch_rewards = reward_tensor[i:i+self.batch_size]
                
                # Train step with policy gradient
                loss = self.train_step_from_evaluation(batch_images, batch_trajectories, batch_rewards)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'average_loss': avg_loss,
            'total_loss': total_loss,
            'num_batches': num_batches,
            'num_epochs': self.num_epochs,
            'avg_reward': torch.mean(reward_tensor).item(),
            'max_reward': torch.max(reward_tensor).item(),
            'min_reward': torch.min(reward_tensor).item()
        }

    def collect_trajectory_for_training(self, simulator, renderer, evaluator, steps: int = 50) -> List[dict]:
        """
        Collect a trajectory by running the current policy and getting evaluation feedback.
        
        :param simulator: Simulator instance
        :param renderer: Renderer instance  
        :param evaluator: Evaluator instance
        :param steps: Number of steps to collect
        :return: List of trajectory data points
        """
        trajectory_data = []
        trajectory_history = []
        state_history = []
        
        # Initialize if needed
        if not hasattr(self, '_initialization'):
            self.initialize(None)
        
        for step in range(steps):
            # Get current state
            current_state = simulator.get_state()
            state_history.append(current_state)
            
            # Get sensor data
            if step == 0:
                original_state = current_state
                last_state = current_state
            else:
                last_state = prev_state
                
            env_state = renderer.get_sensor_input(original_state, last_state, current_state)
            
            # Get RGB image
            if hasattr(env_state, 'rgb_image') and env_state.rgb_image is not None:
                rgb_image = env_state.rgb_image
                
                # Predict trajectory using the new interface
                trajectory = self.compute_planner_trajectory(env_state, state_history)
                
                # Extract steering angle and acceleration from the trajectory for training data
                # We'll use the first few waypoints to estimate the actions taken
                if len(trajectory.waypoints) >= 2:
                    # Calculate steering angle from trajectory curvature
                    wp1, wp2 = trajectory.waypoints[0], trajectory.waypoints[1]
                    heading_diff = wp2.heading - wp1.heading
                    steering_angle = heading_diff / self.sampling_time  # Approximate steering rate
                    steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))
                    
                    # Calculate acceleration from velocity change
                    current_velocity = math.sqrt(current_state.ego_pos.vehicle_parameters.vx**2 + 
                                               current_state.ego_pos.vehicle_parameters.vy**2)
                    # For simplicity, use a default acceleration towards target speed
                    target_speed = 2.0
                    acceleration = (target_speed - current_velocity) / self.sampling_time
                    acceleration = max(-self.max_acceleration, min(self.max_acceleration, acceleration))
                else:
                    steering_angle = 0.0
                    acceleration = 0.0
                
                # Normalize actions to [-1, 1] range for training
                normalized_steering = steering_angle / self.max_steering_angle
                normalized_accel = acceleration / self.max_acceleration
                
                trajectory_history.append(trajectory)
                simulator.do_action(trajectory)
                
                # Get evaluation score
                if len(trajectory_history) > 1:  # Need some history for evaluation
                    reward = evaluator.compute_cumulative_score(trajectory_history, current_state)
                else:
                    reward = 0.0
                
                # Store trajectory data
                trajectory_data.append({
                    'rgb_image': rgb_image,
                    'trajectory': [normalized_steering, normalized_accel],
                    'reward': reward,
                    'step': step
                })
            
            prev_state = current_state
        
        return trajectory_data

    def train_with_evaluation(self, simulator, renderer, evaluator, num_episodes: int = 10, steps_per_episode: int = 50) -> dict:
        """
        Train the RL planner using evaluation-based rewards.
        
        :param simulator: Simulator instance
        :param renderer: Renderer instance
        :param evaluator: Evaluator instance
        :param num_episodes: Number of training episodes
        :param steps_per_episode: Steps per episode
        :return: Training statistics
        """
        print(f"Training RL planner with evaluation feedback for {num_episodes} episodes...")
        
        all_episode_stats = []
        total_episodes = 0
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            # Collect trajectory data
            trajectory_data = self.collect_trajectory_for_training(
                simulator, renderer, evaluator, steps_per_episode
            )
            
            if len(trajectory_data) == 0:
                print("No trajectory data collected, skipping episode")
                continue
                
            # Train on this episode
            episode_stats = self.train_episode_from_evaluation(trajectory_data)
            all_episode_stats.append(episode_stats)
            total_episodes += 1
            
            # Print episode results
            print(f"  Loss: {episode_stats['average_loss']:.4f}, "
                  f"Avg Reward: {episode_stats['avg_reward']:.2f}, "
                  f"Max Reward: {episode_stats['max_reward']:.2f}")
        
        # Compile overall statistics
        if all_episode_stats:
            overall_stats = {
                'episodes_completed': total_episodes,
                'average_loss': np.mean([s['average_loss'] for s in all_episode_stats]),
                'average_reward': np.mean([s['avg_reward'] for s in all_episode_stats]),
                'max_reward_seen': max([s['max_reward'] for s in all_episode_stats]),
                'min_reward_seen': min([s['min_reward'] for s in all_episode_stats]),
                'final_episode_loss': all_episode_stats[-1]['average_loss'],
                'final_episode_reward': all_episode_stats[-1]['avg_reward']
            }
        else:
            overall_stats = {
                'episodes_completed': 0,
                'average_loss': 0.0,
                'average_reward': 0.0,
                'max_reward_seen': 0.0,
                'min_reward_seen': 0.0,
                'final_episode_loss': 0.0,
                'final_episode_reward': 0.0
            }
        
        print(f"\nTraining completed: {overall_stats}")
        return overall_stats

    # Keep the old methods for backward compatibility
    def train_step(self, rgb_image: torch.Tensor, target_actions: torch.Tensor) -> float:
        """
        Perform a single training step (legacy supervised learning method).
        
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
        
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value

    def train_episode(self, rgb_images: List[np.ndarray], target_actions: List[tuple]) -> dict:
        """
        Train on an episode of data (legacy supervised learning method).
        
        :param rgb_images: List of RGB images
        :param target_actions: List of (steering, acceleration) tuples
        :return: Training statistics
        """
        if len(rgb_images) != len(target_actions):
            raise ValueError("RGB images and target actions must have the same length")
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(rgb_images), self.batch_size):
                # Prepare batch
                batch_images = rgb_images[i:i+self.batch_size]
                batch_targets = target_actions[i:i+self.batch_size]
                
                # Convert to tensors
                image_tensor = self._prepare_image_batch(batch_images)
                target_tensor = torch.tensor(batch_targets, dtype=torch.float32)
                
                # Train step
                loss = self.train_step(image_tensor, target_tensor)
                epoch_loss += loss
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'average_loss': avg_loss,
            'total_loss': total_loss,
            'num_batches': num_batches,
            'num_epochs': self.num_epochs
        }

    def _prepare_image_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Prepare a batch of images for training.
        
        :param images: List of numpy arrays representing RGB images
        :return: Tensor of shape [batch_size, 3, H, W]
        """
        batch_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                tensor = torch.from_numpy(img).float()
            else:
                tensor = torch.tensor(img).float()
            
            # Normalize and convert to CHW format
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            if len(tensor.shape) == 3:
                tensor = tensor.permute(2, 0, 1)  # HWC to CHW
            
            batch_tensors.append(tensor)
        
        return torch.stack(batch_tensors)

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
            
        metrics = {
            'loss': loss.item(),
            'mae': mae.item(),
            'steering_mae': steering_mae.item(),
            'acceleration_mae': accel_mae.item()
        }
        
        self.evaluation_metrics.append(metrics)
        return metrics

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
            },
            'training_losses': self.training_losses,
            'evaluation_metrics': self.evaluation_metrics,
        }, path)
        print(f"Model saved to {path}")

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
        
        # Load training history
        self.training_losses = checkpoint.get('training_losses', [])
        self.evaluation_metrics = checkpoint.get('evaluation_metrics', [])
        
        print(f"Model loaded from {path}")

    @classmethod
    def from_checkpoint(cls, path: str) -> 'ReinforcementLearningPlanner':
        """
        Create a ReinforcementLearningPlanner instance from a saved checkpoint.
        
        :param path: Path to the checkpoint
        :return: ReinforcementLearningPlanner instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        
        # Create instance with loaded config
        planner = cls(**config)
        planner.load_model(path)
        
        return planner

    def get_training_stats(self) -> dict:
        """
        Get training statistics.
        
        :return: Dictionary containing training statistics
        """
        return {
            'num_training_steps': len(self.training_losses),
            'average_loss': np.mean(self.training_losses) if self.training_losses else 0.0,
            'latest_loss': self.training_losses[-1] if self.training_losses else 0.0,
            'num_evaluations': len(self.evaluation_metrics),
            'latest_evaluation': self.evaluation_metrics[-1] if self.evaluation_metrics else {}
        }

    def reset_training_history(self):
        """Reset training and evaluation history."""
        self.training_losses = []
        self.evaluation_metrics = []


# Backward compatibility alias
ReinforemenctPlanner = ReinforcementLearningPlanner

if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    
    # Create planner with realistic parameters
    planner = ReinforcementLearningPlanner(
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
    
    print(f"Planner: {planner.name}")
    print(f"Device: {planner.device}")
    print(f"Observation type: {planner.observation_type()}")
    
    # Create a dummy RGB image
    rgb_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # Create environment state
    env_state = EnvState(
        rgb_image=rgb_image,
        depth=np.zeros((240, 320))
    )
    
    # Create a dummy system state for state_history
    dummy_vehicle_params = VehicleParameters(vx=5.0, vy=0.0, steering_angle=0.0)
    dummy_ego_pos = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=5.0, timestamp=0)
    dummy_ego_pos.vehicle_parameters = dummy_vehicle_params
    
    dummy_system_state = SystemState(
        ego_pos=dummy_ego_pos,
        timestamp=0.0
    )
    state_history = [dummy_system_state]
    
    # Generate trajectory
    trajectory = planner.compute_planner_trajectory(env_state, state_history)
    
    print(f"Generated trajectory with {len(trajectory.waypoints)} waypoints")
    for i, waypoint in enumerate(trajectory.waypoints[:5]):  # Print first 5 waypoints
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
    
    # Show training stats
    stats = planner.get_training_stats()
    print(f"Training stats: {stats}")
