"""
This script serves as the entry point for training a reinforcement learning planner.
It initializes the simulator, environment model, evaluator, and RL planner,
and runs the training for a specified number of episodes.
"""
import os
import math
import numpy as np
from typing import List
from tqdm import tqdm
from monarch.rendering.abstract_renderer import AbstractRenderer
from monarch.rendering.omnire import OmniRe
from monarch.simulator.abstract_simulator import AbstractSimulator
from monarch.simulator.nuplan import NuPlan
from monarch.planning.abstract_planner import AbstractPlanner
from monarch.planning.rl_planner import ReinforcementLearningPlanner

from monarch.utils.path_utils import use_path
from monarch.utils.image_utils import save_rgb_images_to_video


class RLEvaluatorAdapter:
    """
    Adapter class that provides reward signals for RL training.
    This evaluator rewards forward progress, smooth trajectories, and safe driving behavior.
    """
    
    def __init__(self):
        self.previous_scores = []
        self.trajectory_count = 0
        
    def compute_cumulative_score(self, trajectory_history, current_state):
        """
        Compute a reward score based on trajectory quality and safety.
        FIXED: Provide stronger and more meaningful reward signals for RL learning.
        
        :param trajectory_history: List of previous trajectories
        :param current_state: Current system state
        :return: Reward score (higher is better)
        """
        if len(trajectory_history) == 0:
            return 0.0
            
        reward = 0.0
        self.trajectory_count += 1
        
        # Extract current position and velocity
        current_pos = current_state.ego_pos
        
        # FIXED: More significant reward scaling for better learning
        if hasattr(current_pos, 'x') and hasattr(current_pos, 'y'):
            if len(self.previous_scores) > 0:
                # Stronger base reward for maintaining motion
                distance_reward = 10.0  # Increased from 3.0
                reward += distance_reward
        
        # FIXED: More discriminative speed rewards
        if hasattr(current_state.ego_pos, 'vehicle_parameters'):
            vx = current_state.ego_pos.vehicle_parameters.vx
            vy = current_state.ego_pos.vehicle_parameters.vy
            speed = math.sqrt(vx**2 + vy**2)
            
            # More discriminative speed rewards with better scaling
            if 4.0 <= speed <= 8.0:  # Optimal urban driving speeds
                reward += 20.0  # Increased from 8.0
            elif 2.0 <= speed < 4.0:  # Reasonable slow speeds
                reward += 10.0  # Increased from 5.0
            elif 1.0 <= speed < 2.0:  # Very slow but moving
                reward += 5.0   # Increased from 2.0
            elif speed < 0.5:  # Nearly stopped - strong penalty
                reward -= 15.0  # Increased penalty magnitude
            elif speed > 12.0:  # Too fast - moderate penalty
                reward -= 8.0   # Increased penalty magnitude
        
        # Reward smooth trajectories (analyze waypoint consistency)
        latest_trajectory = trajectory_history[-1]
        if len(latest_trajectory.waypoints) >= 3:
            smoothness_reward = self._evaluate_trajectory_smoothness(latest_trajectory)
            reward += smoothness_reward * 2.0  # Amplify smoothness rewards
        
        # Safety reward (penalize erratic behavior)
        safety_reward = self._evaluate_trajectory_safety(latest_trajectory, current_state)
        reward += safety_reward * 2.0  # Amplify safety rewards
        
        # FIXED: Add trajectory consistency reward
        if len(trajectory_history) >= 2:
            consistency_reward = self._evaluate_trajectory_consistency(trajectory_history[-2:])
            reward += consistency_reward
        
        # FIXED: Add more diverse reward signals based on trajectory step
        step_bonus = 0.0
        if self.trajectory_count > 0:
            # Progressive reward for longer successful trajectories
            if self.trajectory_count % 10 == 0:  # Every 10 steps
                step_bonus += 5.0
            if self.trajectory_count % 50 == 0:  # Every 50 steps  
                step_bonus += 15.0
        
        reward += step_bonus
        
        # Progress tracking with momentum
        self.previous_scores.append(reward)
        if len(self.previous_scores) > 50:  # Keep recent history
            self.previous_scores = self.previous_scores[-25:]
        
        # FIXED: Add reward variance to help learning
        # Ensure we have some diversity in rewards within an episode
        if len(self.previous_scores) >= 5:
            recent_variance = np.var(self.previous_scores[-5:])
            if recent_variance < 1.0:  # If rewards are too similar
                # Add some performance-based variation
                performance_factor = min(reward / 30.0, 1.0)  # Normalize to [0,1]
                reward += performance_factor * 10.0  # Add up to 10 bonus for good performance
        
        # FIXED: Remove random noise - it hurts learning signal quality
        # reward += np.random.normal(0, 0.1)  # Removed
        
        # Debug: Print reward components occasionally
        if self.trajectory_count % 20 == 0:
            print(f"  Reward breakdown - Total: {reward:.2f}, Base: {distance_reward if 'distance_reward' in locals() else 0:.1f}, "
                  f"Speed component: {20.0 if 4.0 <= speed <= 8.0 else 0:.1f}, Step bonus: {step_bonus:.1f}")
        
        return reward
    
    def _evaluate_trajectory_smoothness(self, trajectory):
        """
        Evaluate trajectory smoothness based on heading changes and waypoint spacing.
        Rebalanced for better learning signals.
        """
        if len(trajectory.waypoints) < 3:
            return 0.0
            
        smoothness_score = 0.0
        
        # Check heading consistency
        heading_changes = []
        for i in range(1, len(trajectory.waypoints)):
            prev_heading = trajectory.waypoints[i-1].heading
            curr_heading = trajectory.waypoints[i].heading
            heading_diff = abs(curr_heading - prev_heading)
            
            # Normalize heading difference to [0, pi]
            if heading_diff > math.pi:
                heading_diff = 2 * math.pi - heading_diff
                
            heading_changes.append(heading_diff)
        
        # More nuanced smoothness rewards
        avg_heading_change = np.mean(heading_changes)
        if avg_heading_change < 0.05:  # Extremely smooth
            smoothness_score += 4.0  # Increased reward
        elif avg_heading_change < 0.1:  # Very smooth
            smoothness_score += 2.5  # Slightly reduced from 3.0
        elif avg_heading_change < 0.2:  # Moderately smooth
            smoothness_score += 1.5  # Increased from 1.0
        elif avg_heading_change < 0.4:  # Acceptable
            smoothness_score += 0.5  # Small positive reward
        else:  # Too jerky
            smoothness_score -= 1.5  # Reduced penalty from -2.0
            
        return smoothness_score
    
    def _evaluate_trajectory_safety(self, trajectory, current_state):
        """
        Evaluate trajectory safety based on reasonable vehicle dynamics.
        Rebalanced to encourage good planning behavior.
        """
        if len(trajectory.waypoints) < 2:
            return 0.0
            
        safety_score = 0.0
        
        # Check for reasonable waypoint spacing
        distances = []
        for i in range(1, len(trajectory.waypoints)):
            wp1 = trajectory.waypoints[i-1]
            wp2 = trajectory.waypoints[i]
            distance = math.sqrt((wp2.x - wp1.x)**2 + (wp2.y - wp1.y)**2)
            distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            # More nuanced waypoint spacing rewards
            if 0.3 <= avg_distance <= 1.5:  # Optimal spacing for 0.1s sampling
                safety_score += 3.0  # Increased from 2.0
            elif 0.1 <= avg_distance < 0.3:  # Acceptable but closer spacing
                safety_score += 1.5  # New reward tier
            elif 1.5 < avg_distance <= 2.5:  # Acceptable but wider spacing
                safety_score += 1.0  # New reward tier
            else:  # Poor spacing
                safety_score -= 0.5  # Reduced penalty from -1.0
        
        # Enhanced path efficiency evaluation
        if len(trajectory.waypoints) >= 3:
            total_path_length = sum(distances) if distances else 0.0
            start_to_end_distance = math.sqrt(
                (trajectory.waypoints[-1].x - trajectory.waypoints[0].x)**2 + 
                (trajectory.waypoints[-1].y - trajectory.waypoints[0].y)**2
            )
            
            if total_path_length > 0:
                efficiency_ratio = start_to_end_distance / total_path_length
                if efficiency_ratio > 0.9:  # Very direct path
                    safety_score += 2.5  # Increased from 1.0
                elif efficiency_ratio > 0.8:  # Relatively straight path
                    safety_score += 1.5  # Increased from 1.0
                elif efficiency_ratio > 0.6:  # Acceptable path
                    safety_score += 0.5  # New reward tier
                elif efficiency_ratio < 0.3:  # Very curved path
                    safety_score -= 0.5  # Reduced penalty from -1.0
        
        return safety_score
    
    def _evaluate_trajectory_consistency(self, trajectory_history):
        """
        Evaluate consistency between consecutive trajectories.
        Rewards smooth transitions and penalizes erratic changes in planning.
        
        :param trajectory_history: List of last 2 trajectories
        :return: Consistency reward score
        """
        if len(trajectory_history) < 2:
            return 0.0
            
        prev_traj = trajectory_history[0]
        curr_traj = trajectory_history[1]
        
        if len(prev_traj.waypoints) < 5 or len(curr_traj.waypoints) < 5:
            return 0.0
        
        consistency_score = 0.0
        
        # Compare the intended future positions from previous trajectory 
        # with the actual path taken in current trajectory
        # Look at waypoints 2-4 steps ahead (0.2-0.4 seconds future planning)
        for i in range(2, min(5, len(prev_traj.waypoints), len(curr_traj.waypoints))):
            prev_wp = prev_traj.waypoints[i]
            curr_wp = curr_traj.waypoints[1]  # Current starting point
            
            # Calculate how much the plan deviated
            predicted_x = prev_wp.x
            predicted_y = prev_wp.y
            actual_x = curr_wp.x
            actual_y = curr_wp.y
            
            deviation = math.sqrt((predicted_x - actual_x)**2 + (predicted_y - actual_y)**2)
            
            # Reward small deviations (consistent planning)
            if deviation < 0.5:  # Very consistent
                consistency_score += 3.0
            elif deviation < 1.0:  # Moderately consistent
                consistency_score += 1.5
            elif deviation < 2.0:  # Acceptable deviation
                consistency_score += 0.5
            else:  # Large deviation - penalize
                consistency_score -= 1.0
        
        # Compare heading consistency
        if len(prev_traj.waypoints) >= 3 and len(curr_traj.waypoints) >= 3:
            prev_heading_change = abs(prev_traj.waypoints[2].heading - prev_traj.waypoints[0].heading)
            curr_heading_change = abs(curr_traj.waypoints[2].heading - curr_traj.waypoints[0].heading)
            
            # Normalize heading differences
            prev_heading_change = min(prev_heading_change, 2*math.pi - prev_heading_change)
            curr_heading_change = min(curr_heading_change, 2*math.pi - curr_heading_change)
            
            heading_consistency = abs(prev_heading_change - curr_heading_change)
            
            if heading_consistency < 0.1:  # Very consistent heading plans
                consistency_score += 2.0
            elif heading_consistency < 0.3:  # Moderately consistent
                consistency_score += 1.0
            elif heading_consistency > 0.8:  # Very inconsistent
                consistency_score -= 2.0
        
        return consistency_score


def train_rl_planner(
    num_episodes: int,
    steps_per_episode: int,
    simulator: AbstractSimulator,
    renderer: AbstractRenderer,
    planner: ReinforcementLearningPlanner,
    evaluator: RLEvaluatorAdapter,
    save_model_path: str = None
):
    """
    Train the RL planner using the evaluation-based training method.
    Handles episode resets at this level to ensure correct path context.
    
    :param num_episodes: Number of training episodes
    :param steps_per_episode: Steps per episode
    :param simulator: The simulator instance
    :param renderer: The renderer instance (environment)
    :param planner: The RL planner instance
    :param evaluator: The evaluator instance
    :param save_model_path: Path to save the trained model
    :return: Training statistics
    """
    print(f"Starting RL training for {num_episodes} episodes with {steps_per_episode} steps each")
    
    # Initialize planner
    planner.initialize(None)
    
    # Store renderer paths for reset operations
    renderer_config_path = None
    renderer_checkpoint_path = None
    if hasattr(renderer, '_config_path'):
        renderer_config_path = renderer._config_path
    if hasattr(renderer, '_checkpoint_path'):
        renderer_checkpoint_path = renderer._checkpoint_path
    
    all_episode_stats = []
    total_episodes = 0
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        # Reset both simulator and renderer for each new episode
        # Handle renderer reset in the correct path context
        try:
            print(f"  Resetting simulator and renderer for episode {episode + 1}")
            simulator.reset()
            
            # Reset renderer in correct path context
            with use_path("./drivestudio", True):
                renderer.reset()
            
            print(f"  Reset completed successfully")
        except Exception as e:
            print(f"  Warning: Reset failed: {e}")
            # Continue anyway - some implementations might not have reset methods
        
        # Collect trajectory data using the planner's method
        trajectory_data = planner.collect_trajectory_for_training(
            simulator, renderer, evaluator, steps_per_episode
        )
        
        if len(trajectory_data) == 0:
            print("No trajectory data collected, skipping episode")
            continue
            
        # Train on this episode
        episode_stats = planner.train_episode_from_evaluation(trajectory_data)
        all_episode_stats.append(episode_stats)
        total_episodes += 1
        
        # FIXED: Add debugging for reward diversity and advantages
        if len(trajectory_data) > 0:
            rewards = [d['reward'] for d in trajectory_data]
            reward_std = np.std(rewards)
            reward_range = max(rewards) - min(rewards)
            print(f"  Episode reward diversity - Std: {reward_std:.2f}, Range: {reward_range:.2f}")
            
            # Warn if rewards are too uniform (will lead to zero advantages)
            if reward_std < 1.0:
                print(f"  WARNING: Low reward diversity may cause zero advantages!")
        
        # Print episode results
        print(f"  Loss: {episode_stats['average_loss']:.4f}, "
              f"Avg Reward: {episode_stats['avg_reward']:.2f}, "
              f"Max Reward: {episode_stats['max_reward']:.2f}")
        
        # Print detailed debug info every 5 episodes
        if episode % 5 == 0:
            if hasattr(planner, 'debug_metrics') and planner.debug_metrics:
                latest_debug = planner.debug_metrics[-1]
                print(f"  Debug - Policy Loss: {latest_debug['policy_loss']:.6f}, "
                      f"Advantages: {latest_debug['mean_advantage']:.6f}, "
                      f"Policy Std: {latest_debug['policy_std']:.4f}, "
                      f"Baseline: {latest_debug['reward_baseline']:.2f}")
    
    # Compile overall statistics
    if all_episode_stats:
        training_stats = {
            'episodes_completed': total_episodes,
            'average_loss': np.mean([s['average_loss'] for s in all_episode_stats]),
            'average_reward': np.mean([s['avg_reward'] for s in all_episode_stats]),
            'max_reward_seen': max([s['max_reward'] for s in all_episode_stats]),
            'min_reward_seen': min([s['min_reward'] for s in all_episode_stats]),
            'final_episode_loss': all_episode_stats[-1]['average_loss'],
            'final_episode_reward': all_episode_stats[-1]['avg_reward']
        }
    else:
        training_stats = {
            'episodes_completed': 0,
            'average_loss': 0.0,
            'average_reward': 0.0,
            'max_reward_seen': 0.0,
            'min_reward_seen': 0.0,
            'final_episode_loss': 0.0,
            'final_episode_reward': 0.0
        }
    
    # Save the trained model if path is provided
    if save_model_path:
        planner.save_model(save_model_path)
        print(f"Model saved to {save_model_path}")
    
    print(f"\nTraining completed: {training_stats}")
    return training_stats


def test_trained_planner(
    n_steps: int,
    simulator: AbstractSimulator,
    renderer: AbstractRenderer,
    planner: ReinforcementLearningPlanner
):
    """
    Test the trained planner by running a simulation and collecting sensor inputs.
    
    :param n_steps: Number of simulation steps
    :param simulator: The simulator instance
    :param renderer: The renderer instance
    :param planner: The trained RL planner
    :return: List of sensor inputs for video generation
    """
    print(f"Testing trained planner for {n_steps} steps")
    
    # Reset both simulator and renderer in correct path context
    simulator.reset()
    with use_path("./drivestudio", True):
        renderer.reset()
    
    sensor_inputs = []
    original_state = simulator.get_state()
    state_history = [original_state]
    current_state = original_state
    
    for i in tqdm(range(n_steps), desc="Testing"):
        last_state = current_state
        current_state = simulator.get_state()
        state_history.append(current_state)
        
        # Get sensor input
        sensor_input = renderer.get_sensor_input(original_state, last_state, current_state, True)
        sensor_inputs.append(sensor_input)
        
        # Generate trajectory using trained planner
        trajectory = planner.compute_planner_trajectory(sensor_input, state_history)
        
        # Execute the trajectory
        simulator.do_action(trajectory)
    
    return sensor_inputs


def main():
    """Main function to run RL training and testing."""
    
    renderer = None
    
    # Initialize environment renderer
    with use_path("./drivestudio", True):
        relative_config_path = "configs/datasets/nuplan/8cams_undistorted.yaml"
        relative_checkpoint_path = "output/master-project/run_final"

        if not os.path.exists(relative_config_path):
            print(f"ERROR: Config file not found at {os.path.abspath(relative_config_path)}")
        if not os.path.exists(relative_checkpoint_path):
            print(f"ERROR: Checkpoint directory not found at {os.path.abspath(relative_checkpoint_path)}")

        if os.path.exists(relative_config_path) and os.path.exists(relative_checkpoint_path):
            renderer = OmniRe(relative_config_path, relative_checkpoint_path)
            print("Successfully initialized OmniRe environment model")
        else:
            print("Failed to initialize environment model due to missing files")
            return

    # Initialize simulator
    simulator = NuPlan("2021.05.12.22.00.38_veh-35_01008_01518")
    
    # Initialize RL planner with appropriate parameters
    planner = ReinforcementLearningPlanner(
        hidden_dimensions=[32, 64, 128],
        horizon_seconds=3.0,
        sampling_time=0.1,
        base_velocity=5.0,
        max_steering_angle=math.pi/3,  # 30 degrees
        max_acceleration=5.0,  # 2 m/s²
        image_dimension=(360, 640),  # Should match renderer output
        linear_width=256,
        batch_size=8,  # Increased from 4 for better advantage estimation
        num_epochs=2,  # Reduced from 4 to prevent overfitting on small datasets
        lr=3e-4,       # Reduced learning rate for more stable training
        optimizer_type="Adam"
    )
    
    # Initialize evaluator
    evaluator = RLEvaluatorAdapter()
    
    # Training parameters
    num_episodes = 15  # Reduced from 20, but with better reward signals
    steps_per_episode = 150  # Reduced from 200 to focus on quality over quantity
    save_model_path = "trained_rl_planner.pth"
    
    try:
        # Run training
        training_stats = train_rl_planner(
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            simulator=simulator,
            renderer=renderer,
            planner=planner,
            evaluator=evaluator,
            save_model_path=save_model_path
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Final training statistics: {training_stats}")
        
        # Test the trained planner
        print("\nTesting the trained planner...")
        test_steps = 299
        sensor_inputs = test_trained_planner(test_steps, simulator, renderer, planner)
        
        # Save video of the test run
        save_rgb_images_to_video(sensor_inputs, "trained_planner_test.mp4", brightness=1.0, gamma=1.2)
        print("Test video saved as 'trained_planner_test.mp4'")
        
        # Print final statistics
        final_stats = planner.get_training_stats()
        print(f"\nFinal planner statistics: {final_stats}")
        
        # Print detailed training summary
        planner.print_training_summary()
        
        # Plot training progress
        print("\nGenerating training plots...")
        planner.plot_training_progress(save_path="training_progress.png", show_plot=False)
        print("Training plots saved as 'training_progress.png'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Even if training failed, try to plot what we have
        if 'planner' in locals() and hasattr(planner, 'training_losses') and planner.training_losses:
            print("\nAttempting to plot partial training data...")
            try:
                planner.plot_training_progress(save_path="partial_training_progress.png", show_plot=False)
                planner.print_training_summary()
            except Exception as plot_e:
                print(f"Failed to plot training data: {plot_e}")

if __name__ == "__main__":
    main()