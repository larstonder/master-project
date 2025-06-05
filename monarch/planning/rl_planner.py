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
        # Get all trainable parameters
        params = list(self.model.parameters())
        if hasattr(self, 'log_std'):
            params.append(self.log_std)
            
        # FIXED: Increase learning rate for faster learning 
        effective_lr = self.lr * 2.0  # Double the learning rate
            
        if self.optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(params, lr=effective_lr)
        elif self.optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=effective_lr, momentum=0.9)
        elif self.optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=effective_lr)
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
        
        # Forward pass to get predicted actions (policy means)
        predicted_actions = self.model(rgb_image)
        
        # Use a learnable standard deviation for better exploration
        if not hasattr(self, 'log_std'):
            self.log_std = nn.Parameter(torch.zeros(self.output_dimension).to(self.device))
        
        policy_std = torch.exp(self.log_std).clamp(min=0.02, max=0.2)  # FIXED: Much lower max std
        
        # Calculate log probabilities using Gaussian policy
        # log π(a|s) = -0.5 * [(a-μ)²/σ² + log(2πσ²)]
        action_diff = actions_taken - predicted_actions
        log_probs = -0.5 * (
            torch.sum((action_diff / policy_std) ** 2, dim=1) +
            2 * torch.sum(torch.log(policy_std)) + 
            actions_taken.shape[1] * np.log(2 * np.pi)
        )
        
        # FIXED: Better advantage calculation using running mean baseline
        if not hasattr(self, 'reward_baseline'):
            self.reward_baseline = 0.0
            self.baseline_alpha = 0.1
        
        # Update running baseline
        current_reward_mean = torch.mean(rewards).item()
        self.reward_baseline = (1 - self.baseline_alpha) * self.reward_baseline + self.baseline_alpha * current_reward_mean
        
        # Calculate advantages using running baseline
        advantages = rewards - self.reward_baseline
        
        # FIXED: Only normalize variance if it's genuinely large, and preserve more signal
        original_advantages = advantages.clone()
        if len(advantages) > 1:
            adv_std = torch.std(advantages)
            if adv_std > 2.0:  # Only normalize if std is quite large
                advantages = advantages / adv_std
                print(f"  Normalizing advantages: original_std={adv_std:.3f}")
            else:
                print(f"  Keeping raw advantages: std={adv_std:.3f}")
        
        # FIXED: Only proceed with training if we have meaningful advantages
        advantage_magnitude = torch.mean(torch.abs(original_advantages)).item()  # Use original, not normalized
        if advantage_magnitude < 0.1:  # Increased threshold
            print(f"Warning: Advantages too small ({advantage_magnitude:.3f}), skipping update")
            return 0.0
        
        # Policy gradient loss: maximize E[log π(a|s) * A(s,a)]
        # In PyTorch we minimize, so negate: -E[log π(a|s) * A(s,a)]
        policy_loss = -torch.mean(log_probs * advantages)
        
        # FIXED: Reduce entropy weight - it was dominating the loss
        # Entropy for multivariate Gaussian: H = 0.5 * log((2πe)^k * |Σ|)
        # For diagonal covariance: H = 0.5 * k * log(2πe) + 0.5 * sum(log(σ²))
        entropy = 0.5 * (actions_taken.shape[1] * np.log(2 * np.pi * np.e) + torch.sum(2 * torch.log(policy_std)))
        entropy_loss = -0.001 * entropy  # FIXED: Reduced from 0.01 to 0.001
        
        # FIXED: Remove the value function loss - it doesn't make sense here
        # The value function should be a separate network, not the policy network
        
        # Total loss: policy loss + entropy loss
        total_loss = policy_loss + entropy_loss
        
        # FIXED: Clip loss to prevent extreme values that destabilize training
        total_loss = torch.clamp(total_loss, min=-2.0, max=2.0)
        
        # FIXED: Add gradient clipping before backward pass and check for NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid loss detected ({total_loss.item()}), skipping update")
            return 0.0
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if hasattr(self, 'log_std'):
            torch.nn.utils.clip_grad_norm_([self.log_std], max_norm=1.0)
        
        self.optimizer.step()
        
        loss_value = total_loss.item()
        self.training_losses.append(loss_value)
        
        # Store additional metrics for debugging
        if not hasattr(self, 'debug_metrics'):
            self.debug_metrics = []
        
        # FIXED: Add more comprehensive debugging information
        with torch.no_grad():
            action_std = torch.std(predicted_actions, dim=0)
            reward_spread = torch.max(rewards) - torch.min(rewards)
        
        self.debug_metrics.append({
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.item(),
            'mean_reward': torch.mean(rewards).item(),
            'mean_advantage': torch.mean(advantages).item(),
            'mean_advantage_original': torch.mean(original_advantages).item(),
            'advantage_std': torch.std(advantages).item(),
            'advantage_std_original': torch.std(original_advantages).item(),
            'advantage_magnitude': advantage_magnitude,
            'mean_log_prob': torch.mean(log_probs).item(),
            'policy_std': torch.mean(policy_std).item(),
            'reward_baseline': self.reward_baseline,
            'reward_spread': reward_spread.item(),
            'action_diversity_steering': action_std[0].item(),
            'action_diversity_accel': action_std[1].item()
        })
        
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
        
        # FIXED: Improve training by using the entire episode as one batch when possible
        # This helps with advantage estimation and reduces noise
        total_loss = 0.0
        num_batches = 0
        
        # Use larger effective batch size by accumulating gradients if needed
        effective_batch_size = max(self.batch_size, min(16, len(trajectory_data)))
        
        for epoch in range(self.num_epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(len(trajectory_data))
            
            for i in range(0, len(trajectory_data), effective_batch_size):
                # Prepare batch with shuffled indices
                batch_indices = indices[i:i+effective_batch_size]
                batch_images = image_tensor[batch_indices]
                batch_trajectories = trajectory_tensor[batch_indices]
                batch_rewards = reward_tensor[batch_indices]
                
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
            'min_reward': torch.min(reward_tensor).item(),
            'effective_batch_size': effective_batch_size
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
                
                # Get the action that the model would predict for this state
                steering_angle, acceleration = self._predict_actions(rgb_image)
                
                # Predict trajectory using the new interface
                trajectory = self.compute_planner_trajectory(env_state, state_history)
                
                # Store the actual actions taken (normalized to [-1, 1])
                normalized_steering = steering_angle / self.max_steering_angle
                normalized_accel = acceleration / self.max_acceleration
                
                # Clamp to ensure actions are in valid range
                normalized_steering = max(-1.0, min(1.0, normalized_steering))
                normalized_accel = max(-1.0, min(1.0, normalized_accel))
                
                trajectory_history.append(trajectory)
                simulator.do_action(trajectory)
                
                # Get evaluation score based on the trajectory taken
                if len(trajectory_history) >= 1:  # Can evaluate from first trajectory
                    reward = evaluator.compute_cumulative_score(trajectory_history, current_state)
                else:
                    reward = 0.0
                
                # Store trajectory data with consistent action representation
                trajectory_data.append({
                    'rgb_image': rgb_image,
                    'trajectory': [normalized_steering, normalized_accel],  # Actions that were taken
                    'reward': reward,
                    'step': step,
                    'raw_steering': steering_angle,  # For debugging
                    'raw_acceleration': acceleration  # For debugging
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

    def plot_training_progress(self, save_path: str = None, show_plot: bool = True):
        """
        Plot training loss and metrics over time.
        
        :param save_path: Path to save the plot (optional)
        :param show_plot: Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Install with: pip install matplotlib")
            return
        
        if not self.training_losses:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Progress', fontsize=16)
        
        # Plot 1: Training Loss
        axes[0, 0].plot(self.training_losses, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average for loss
        if len(self.training_losses) > 10:
            window_size = min(50, len(self.training_losses) // 5)
            moving_avg = []
            for i in range(len(self.training_losses)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(self.training_losses[start_idx:i+1]))
            axes[0, 0].plot(moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[0, 0].legend()
        
        # Plot 2: Reward Distribution (if debug metrics available)
        if hasattr(self, 'debug_metrics') and self.debug_metrics:
            rewards = [m['mean_reward'] for m in self.debug_metrics]
            axes[0, 1].plot(rewards, 'g-', alpha=0.7)
            axes[0, 1].set_title('Mean Rewards')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Mean Reward')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Policy loss
            policy_losses = [m['policy_loss'] for m in self.debug_metrics]
            axes[1, 0].plot(policy_losses, 'orange', alpha=0.7)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Policy Loss')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Advantages
            advantages = [m['mean_advantage'] for m in self.debug_metrics]
            axes[1, 1].plot(advantages, 'purple', alpha=0.7)
            axes[1, 1].set_title('Mean Advantages')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Mean Advantage')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        else:
            # If no debug metrics, plot loss statistics
            if len(self.training_losses) > 1:
                # Loss variance
                loss_var = []
                window = 20
                for i in range(window, len(self.training_losses)):
                    loss_var.append(np.var(self.training_losses[i-window:i]))
                axes[0, 1].plot(range(window, len(self.training_losses)), loss_var, 'orange')
                axes[0, 1].set_title('Loss Variance (window=20)')
                axes[0, 1].set_xlabel('Training Step')
                axes[0, 1].set_ylabel('Loss Variance')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Loss histogram
            axes[1, 0].hist(self.training_losses, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('Loss Distribution')
            axes[1, 0].set_xlabel('Loss Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Loss trends (first/last comparison)
            if len(self.training_losses) > 100:
                first_100 = self.training_losses[:100]
                last_100 = self.training_losses[-100:]
                
                axes[1, 1].boxplot([first_100, last_100], labels=['First 100', 'Last 100'])
                axes[1, 1].set_title('Loss Comparison (First vs Last 100 steps)')
                axes[1, 1].set_ylabel('Loss Value')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Need more data\nfor comparison', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Loss Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig

    def print_training_summary(self):
        """Print a summary of training progress and statistics."""
        if not self.training_losses:
            print("No training data available")
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print(f"Total training steps: {len(self.training_losses)}")
        print(f"Current loss: {self.training_losses[-1]:.4f}")
        print(f"Average loss: {np.mean(self.training_losses):.4f}")
        print(f"Best loss: {min(self.training_losses):.4f}")
        print(f"Worst loss: {max(self.training_losses):.4f}")
        print(f"Loss std: {np.std(self.training_losses):.4f}")
        
        # Check for training trends
        if len(self.training_losses) > 50:
            recent_loss = np.mean(self.training_losses[-50:])
            early_loss = np.mean(self.training_losses[:50])
            improvement = early_loss - recent_loss
            
            print(f"\nTraining Progress:")
            print(f"Early loss (first 50): {early_loss:.4f}")
            print(f"Recent loss (last 50): {recent_loss:.4f}")
            print(f"Improvement: {improvement:.4f} ({'✓' if improvement > 0 else '✗'})")
        
        # FIXED: Debug metrics summary with new structure
        if hasattr(self, 'debug_metrics') and self.debug_metrics:
            print(f"\nDetailed Metrics (last 10 steps):")
            recent_metrics = self.debug_metrics[-10:]
            
            avg_policy_loss = np.mean([m['policy_loss'] for m in recent_metrics])
            avg_entropy_loss = np.mean([m['entropy_loss'] for m in recent_metrics])
            avg_reward = np.mean([m['mean_reward'] for m in recent_metrics])
            avg_advantage = np.mean([m['mean_advantage'] for m in recent_metrics])
            avg_advantage_std = np.mean([m['advantage_std'] for m in recent_metrics])
            avg_advantage_mag = np.mean([m['advantage_magnitude'] for m in recent_metrics])
            avg_policy_std = np.mean([m['policy_std'] for m in recent_metrics])
            avg_reward_spread = np.mean([m['reward_spread'] for m in recent_metrics])
            avg_action_div_steer = np.mean([m['action_diversity_steering'] for m in recent_metrics])
            avg_action_div_accel = np.mean([m['action_diversity_accel'] for m in recent_metrics])
            final_baseline = recent_metrics[-1]['reward_baseline']
            
            print(f"Avg Policy Loss: {avg_policy_loss:.4f}")
            print(f"Avg Entropy Loss: {avg_entropy_loss:.4f}")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Avg Advantage: {avg_advantage:.4f}")
            print(f"Avg Advantage Std: {avg_advantage_std:.4f}")
            print(f"Avg Advantage Magnitude: {avg_advantage_mag:.6f}")
            print(f"Avg Policy Std: {avg_policy_std:.4f}")
            print(f"Avg Reward Spread: {avg_reward_spread:.2f}")
            print(f"Action Diversity (Steer/Accel): {avg_action_div_steer:.4f} / {avg_action_div_accel:.4f}")
            print(f"Reward Baseline: {final_baseline:.2f}")
        
        print("="*60)


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
