#!/usr/bin/env python3
"""
Example script demonstrating RL planner integration with simulator, environment, and evaluator.
This script can be run independently to test the RL planner functionality.
"""

import os
import sys
import time
import numpy as np
import torch

# Add the monarch directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planning.rl_planner import ReinforcementLearningPlanner
from types.state_types import SystemState, VehicleState, EnvState
from types.action import Action, Trajectory, Waypoint
from simulator.abstract_simulator import AbstractSimulator
from environment.abstract_environment import AbstractEnvironment
from evaluation.abstract_evaluator import AbstractEvaluator


class SimpleSimulator(AbstractSimulator):
    """Simple simulator for demonstration."""
    
    def __init__(self):
        self.time = 0.0
        self.ego = VehicleState(0.0, 0.0, 0.0, 0.0, 0)
        self.step_count = 0
    
    def get_state(self) -> SystemState:
        """Get current state."""
        # Add some variation to make it interesting
        noise_x = 0.1 * np.sin(self.time)
        noise_y = 0.1 * np.cos(self.time)
        
        other_vehicles = [
            VehicleState(5.0 + noise_x, 2.0 + noise_y, 0.1, 0.0, 1),
            VehicleState(-3.0 - noise_x, -1.0 - noise_y, -0.1, 0.0, 2)
        ]
        
        return SystemState(
            ego_pos=self.ego,
            vehicle_pos_list=other_vehicles,
            timestamp=self.time
        )
    
    def do_action(self, action: Action):
        """Execute action."""
        if action and action.trajectory and len(action.trajectory.waypoints) > 1:
            # Move to next waypoint
            next_wp = action.trajectory.waypoints[1]
            self.ego.x = next_wp.x
            self.ego.y = next_wp.y 
            self.ego.heading = next_wp.heading
        
        self.time += 0.1
        self.step_count += 1


class SimpleEnvironment(AbstractEnvironment):
    """Simple environment that generates RGB images."""
    
    def __init__(self, image_size=(240, 320)):
        self.image_size = image_size
        
    def get_sensor_input(self, original_state, last_state, current_state) -> EnvState:
        """Generate sensor output with varying RGB patterns."""
        # Create a simple pattern that changes based on ego position
        x, y = current_state.ego_pos.x, current_state.ego_pos.y
        
        # Create RGB image with patterns based on position
        rgb = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Add some spatial patterns
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                # Red channel varies with x position
                rgb[i, j, 0] = int(128 + 64 * np.sin(x + i/50.0))
                # Green channel varies with y position  
                rgb[i, j, 1] = int(128 + 64 * np.cos(y + j/50.0))
                # Blue channel varies with both
                rgb[i, j, 2] = int(128 + 32 * np.sin(x + y + (i+j)/100.0))
        
        # Clamp values
        rgb = np.clip(rgb, 0, 255)
        
        # Simple depth (distance from center)
        depth = np.zeros(self.image_size, dtype=np.float32)
        center_i, center_j = self.image_size[0]//2, self.image_size[1]//2
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                depth[i, j] = np.sqrt((i-center_i)**2 + (j-center_j)**2) / 100.0
        
        env_state = EnvState(rgb_image=rgb, depth=depth)
        env_state.timestamp = current_state.timestamp
        return env_state


class SimpleMetric:
    """Simple metric for evaluation."""
    
    def __init__(self, name):
        self.name = name
        
    def compute(self, history, scenario):
        """Compute a simple metric based on trajectory smoothness."""
        if len(history) < 2:
            return 0.0
            
        # Calculate trajectory smoothness (lower is better)
        total_curvature = 0.0
        for i in range(1, len(history)):
            action = history[i]
            if action and action.trajectory and len(action.trajectory.waypoints) > 2:
                waypoints = action.trajectory.waypoints
                for j in range(1, len(waypoints)-1):
                    # Calculate angle change
                    angle1 = waypoints[j-1].heading
                    angle2 = waypoints[j].heading
                    angle3 = waypoints[j+1].heading
                    
                    curvature = abs(angle3 - 2*angle2 + angle1)
                    total_curvature += curvature
        
        # Return inverse of curvature as score (higher is better)
        return max(0.0, 10.0 - total_curvature)


class SimpleEvaluator(AbstractEvaluator):
    """Simple evaluator."""
    
    def __init__(self):
        self._metrics = [
            SimpleMetric("trajectory_smoothness"),
            SimpleMetric("path_efficiency")
        ]
    
    @property
    def name(self):
        return "SimpleEvaluator"
    
    @property 
    def metrics(self):
        return self._metrics
        
    def compute_cumulative_score(self, history, scenario):
        """Compute cumulative score."""
        total_score = 0.0
        for metric in self._metrics:
            score = metric.compute(history, scenario)
            total_score += score
        return total_score


def train_rl_planner_demo():
    """Demonstrate training the RL planner with evaluation-based rewards."""
    print("="*60)
    print("RL PLANNER EVALUATION-BASED TRAINING DEMONSTRATION")
    print("="*60)
    
    # Create RL planner
    planner = ReinforcementLearningPlanner(
        hidden_dimensions=[16, 32, 64],
        horizon_seconds=2.0,
        sampling_time=0.1,
        base_velocity=3.0,
        max_steering_angle=np.pi/4,
        max_acceleration=2.0,
        image_dimension=(240, 320),
        linear_width=128,
        batch_size=4,
        num_epochs=2,
        lr=1e-3
    )
    
    print(f"Created {planner.name}")
    print(f"Model parameters: {sum(p.numel() for p in planner.model.parameters()):,}")
    print(f"Device: {planner.device}")
    
    # Create components for training
    simulator = SimpleSimulator()
    environment = SimpleEnvironment()
    evaluator = SimpleEvaluator()
    
    print("\nTraining with evaluation-based rewards...")
    start_time = time.time()
    
    # Train using evaluation feedback
    training_stats = planner.train_with_evaluation(
        simulator=simulator,
        environment=environment, 
        evaluator=evaluator,
        num_episodes=5,  # Fewer episodes for demo
        steps_per_episode=20
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Training stats: {training_stats}")
    
    return planner


def compare_training_methods():
    """Compare traditional supervised learning vs evaluation-based training."""
    print("="*60)
    print("TRAINING METHODS COMPARISON")
    print("="*60)
    
    # Create two identical planners
    planner_supervised = ReinforcementLearningPlanner(
        hidden_dimensions=[16, 32],
        horizon_seconds=1.0,
        sampling_time=0.1,
        base_velocity=3.0,
        image_dimension=(240, 320),
        linear_width=64,
        lr=1e-3
    )
    
    planner_evaluation = ReinforcementLearningPlanner(
        hidden_dimensions=[16, 32],
        horizon_seconds=1.0,
        sampling_time=0.1,
        base_velocity=3.0,
        image_dimension=(240, 320),
        linear_width=64,
        lr=1e-3
    )
    
    # Traditional supervised learning training
    print("\n1. Traditional Supervised Learning:")
    print("-" * 40)
    
    # Generate synthetic training data
    num_samples = 50
    rgb_images = []
    target_actions = []
    
    env = SimpleEnvironment()
    
    for i in range(num_samples):
        ego_pos = VehicleState(
            x=np.random.uniform(-5, 5),
            y=np.random.uniform(-5, 5), 
            z=0.0,
            heading=np.random.uniform(-np.pi, np.pi),
            id=0
        )
        
        state = SystemState(ego_pos, [], i * 0.1)
        env_state = env.get_sensor_input(state, state, state)
        rgb_images.append(env_state.rgb_image)
        
        # Simple target policy: steer toward origin
        dx = 0.0 - ego_pos.x
        dy = 0.0 - ego_pos.y
        target_heading = np.arctan2(dy, dx)
        heading_diff = target_heading - ego_pos.heading
        
        while heading_diff > np.pi:
            heading_diff -= 2*np.pi
        while heading_diff < -np.pi:
            heading_diff += 2*np.pi
            
        steering = np.clip(heading_diff / np.pi, -1.0, 1.0)
        distance = np.sqrt(dx**2 + dy**2)
        acceleration = np.clip(distance / 5.0 - 0.5, -1.0, 1.0)
        
        target_actions.append((steering, acceleration))
    
    # Train supervised
    supervised_stats = planner_supervised.train_episode(rgb_images, target_actions)
    print(f"Supervised learning loss: {supervised_stats['average_loss']:.4f}")
    
    # 2. Evaluation-based training
    print("\n2. Evaluation-Based Training:")
    print("-" * 40)
    
    simulator = SimpleSimulator()
    environment = SimpleEnvironment()
    evaluator = SimpleEvaluator()
    
    evaluation_stats = planner_evaluation.train_with_evaluation(
        simulator=simulator,
        environment=environment,
        evaluator=evaluator,
        num_episodes=3,
        steps_per_episode=15
    )
    
    print(f"Evaluation-based training completed")
    print(f"Average reward: {evaluation_stats['average_reward']:.2f}")
    print(f"Max reward seen: {evaluation_stats['max_reward_seen']:.2f}")
    
    # Test both planners
    print("\n3. Performance Comparison:")
    print("-" * 40)
    
    test_rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    test_env_state = EnvState(rgb_image=test_rgb, depth=np.zeros((240, 320)))
    test_env_state.timestamp = time.time()
    
    # Test supervised planner
    action_supervised = planner_supervised.compute_planner_trajectory(test_env_state)
    print(f"Supervised planner generated {len(action_supervised.trajectory.waypoints)} waypoints")
    
    # Test evaluation-based planner
    action_evaluation = planner_evaluation.compute_planner_trajectory(test_env_state)
    print(f"Evaluation-based planner generated {len(action_evaluation.trajectory.waypoints)} waypoints")
    
    return {
        'supervised_stats': supervised_stats,
        'evaluation_stats': evaluation_stats,
        'supervised_planner': planner_supervised,
        'evaluation_planner': planner_evaluation
    }


def demonstrate_evaluation_feedback():
    """Demonstrate how the planner learns from evaluation feedback."""
    print("="*60)
    print("EVALUATION FEEDBACK LEARNING DEMONSTRATION")
    print("="*60)
    
    # Create planner and components
    planner = ReinforcementLearningPlanner(
        hidden_dimensions=[16, 32],
        horizon_seconds=1.5,
        sampling_time=0.1,
        base_velocity=3.0,
        image_dimension=(240, 320),
        linear_width=64,
        lr=1e-3
    )
    
    simulator = SimpleSimulator()
    environment = SimpleEnvironment()
    evaluator = SimpleEvaluator()
    
    print("\nCollecting trajectory with evaluation feedback...")
    
    # Collect a single trajectory and show the feedback process
    trajectory_data = planner.collect_trajectory_for_training(
        simulator=simulator,
        environment=environment,
        evaluator=evaluator,
        steps=10
    )
    
    print(f"Collected {len(trajectory_data)} trajectory points")
    
    # Show some trajectory data
    for i, data_point in enumerate(trajectory_data[:5]):
        print(f"Step {data_point['step']}: "
              f"Action=[{data_point['action'][0]:.3f}, {data_point['action'][1]:.3f}], "
              f"Reward={data_point['reward']:.2f}")
    
    # Train on this trajectory
    if trajectory_data:
        training_stats = planner.train_episode_from_evaluation(trajectory_data)
        print(f"\nTraining on trajectory:")
        print(f"  Loss: {training_stats['average_loss']:.4f}")
        print(f"  Avg Reward: {training_stats['avg_reward']:.2f}")
        print(f"  Reward Range: [{training_stats['min_reward']:.2f}, {training_stats['max_reward']:.2f}]")
    
    return planner


def run_simulation_demo(planner, steps=30):
    """Run a simulation with the trained planner."""
    print("="*60)
    print("SIMULATION DEMONSTRATION")
    print("="*60)
    
    # Create components
    simulator = SimpleSimulator()
    environment = SimpleEnvironment()
    evaluator = SimpleEvaluator()
    
    # Initialize planner
    planner.initialize(None)
    
    print(f"Running simulation with {planner.name} for {steps} steps...")
    
    # Track results
    trajectory_history = []
    scores = []
    positions = []
    
    for step in range(steps):
        # Get current state
        current_state = simulator.get_state()
        positions.append((current_state.ego_pos.x, current_state.ego_pos.y))
        
        # Get sensor data
        if step == 0:
            original_state = current_state
            last_state = current_state
        else:
            last_state = prev_state
            
        env_state = environment.get_sensor_input(original_state, last_state, current_state)
        
        # Plan trajectory
        start_time = time.time()
        action = planner.compute_trajectory(env_state)
        planning_time = time.time() - start_time
        
        trajectory_history.append(action)
        
        # Evaluate
        if step > 0:
            score = evaluator.compute_cumulative_score(trajectory_history, current_state)
            scores.append(score)
        else:
            scores.append(0.0)
        
        # Execute action
        simulator.do_action(action)
        
        # Print progress
        if step % 5 == 0 or step < 5:
            pos = current_state.ego_pos
            print(f"Step {step:2d}: pos=({pos.x:5.2f},{pos.y:5.2f}) "
                  f"heading={pos.heading:5.2f} score={scores[-1]:5.2f} "
                  f"planning={planning_time*1000:.1f}ms")
        
        prev_state = current_state
    
    # Summary
    final_pos = positions[-1]
    avg_score = np.mean(scores) if scores else 0.0
    total_distance = sum(np.sqrt((positions[i][0]-positions[i-1][0])**2 + 
                                (positions[i][1]-positions[i-1][1])**2) 
                        for i in range(1, len(positions)))
    
    print(f"\nSimulation Summary:")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Total distance: {total_distance:.2f}m")
    print(f"  Average score: {avg_score:.2f}")
    print(f"  Number of waypoints per trajectory: {len(action.trajectory.waypoints) if action else 0}")
    
    return {
        'positions': positions,
        'scores': scores,
        'total_distance': total_distance,
        'avg_score': avg_score
    }


def main():
    """Main demonstration function."""
    print("RL PLANNER INTEGRATION DEMONSTRATION")
    print("="*60)
    
    try:
        # 1. Demonstrate evaluation-based training
        trained_planner = train_rl_planner_demo()
        
        # 2. Run simulation with the evaluation-trained planner
        results = run_simulation_demo(trained_planner, steps=25)
        
        # 3. Compare training methods
        print("\n" + "="*60)
        comparison_results = compare_training_methods()
        
        # 4. Demonstrate evaluation feedback process
        print("\n" + "="*60)
        feedback_planner = demonstrate_evaluation_feedback()
        
        # 5. Save the evaluation-trained model
        model_path = "evaluation_trained_rl_planner.pth"
        trained_planner.save_model(model_path)
        print(f"\nEvaluation-trained model saved to: {model_path}")
        
        # 6. Test loading the model
        print("\nTesting model loading...")
        loaded_planner = ReinforcementLearningPlanner.from_checkpoint(model_path)
        print(f"Successfully loaded model: {loaded_planner.name}")
        
        # 7. Quick test with loaded model
        test_rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        test_env_state = EnvState(rgb_image=test_rgb, depth=np.zeros((240, 320)))
        test_env_state.timestamp = time.time()
        
        test_action = loaded_planner.compute_planner_trajectory(test_env_state)
        print(f"Loaded model generated trajectory with {len(test_action.trajectory.waypoints)} waypoints")
        
        # 8. Show final training statistics
        stats = trained_planner.get_training_stats()
        print(f"\nFinal Training Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*60)
        print("KEY INSIGHTS FROM EVALUATION-BASED TRAINING:")
        print("="*60)
        print("1. The planner learns directly from evaluator feedback")
        print("2. No need for predefined target actions")
        print("3. Policy gradient approach adapts to reward signals")
        print("4. Exploration is encouraged through entropy bonus")
        print("5. Training is based on actual performance metrics")
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'simulation_results': results,
            'comparison_results': comparison_results,
            'trained_planner': trained_planner,
            'feedback_planner': feedback_planner
        }
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main() 