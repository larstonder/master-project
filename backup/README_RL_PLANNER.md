# Reinforcement Learning Planner for Autonomous Vehicles

This document describes the implementation of a Reinforcement Learning (RL) planner that integrates with the simulator, environment, and evaluator components of the autonomous vehicle simulation system.

## Overview

The RL planner (`ReinforcementLearningPlanner`) is a neural network-based planning system that:

1. **Processes RGB images** from the environment using a CNN architecture
2. **Outputs steering and acceleration commands** through a fully connected neural network
3. **Generates realistic trajectories** using bicycle model vehicle dynamics
4. **Supports training and evaluation** with configurable loss functions and optimizers
5. **Integrates seamlessly** with the existing simulator, environment, and evaluator framework

## Architecture

### Neural Network Design

The RL planner uses an efficient CNN architecture inspired by MobileNet:

- **Input**: RGB images (configurable dimensions, default 360x640x3)
- **CNN Backbone**: Depthwise separable convolutions for efficiency
- **Output**: 2D action vector [steering_angle, acceleration] in range [-1, 1]
- **Vehicle Dynamics**: Bicycle model for realistic trajectory generation

### Key Components

```python
class ReinforcementLearningPlanner(AbstractPlanner):
    """
    Main RL planner class that inherits from AbstractPlanner
    """
    
    def __init__(self, hidden_dimensions, horizon_seconds, ...):
        # Configuration for CNN, vehicle dynamics, and training
        
    def compute_planner_trajectory(self, current_input) -> Action:
        # Main planning method that returns trajectory waypoints
        
    def train_step(self, rgb_image, target_actions) -> float:
        # Single training step for supervised learning
        
    def train_episode(self, rgb_images, target_actions) -> dict:
        # Train on a full episode of data
```

## Integration with System Components

### 1. Simulator Integration

The planner works with any simulator implementing `AbstractSimulator`:

```python
# Get current state from simulator
current_state = simulator.get_state()

# Plan trajectory
action = planner.compute_trajectory(sensor_output)

# Execute trajectory in simulator
simulator.do_action(action)
```

### 2. Environment Integration

The planner expects RGB images from the environment:

```python
# Environment provides sensor data including RGB images
sensor_output = environment.get_sensor_input(original_state, last_state, current_state)

# RL planner processes the RGB image
action = planner.compute_planner_trajectory(sensor_output)
```

### 3. Evaluator Integration

The evaluator assesses the planner's performance:

```python
# Evaluator computes metrics on the planned trajectories
score = evaluator.compute_cumulative_score(trajectory_history, scenario)
```

## Usage Examples

### Basic Usage

```python
from planning.rl_planner import ReinforcementLearningPlanner
from monarch.typings.state_types import EnvState
import numpy as np

# Create RL planner
planner = ReinforcementLearningPlanner(
    hidden_dimensions=[32, 64, 128],
    horizon_seconds=3.0,
    sampling_time=0.1,
    base_velocity=5.0,
    max_steering_angle=np.pi/6,  # 30 degrees
    max_acceleration=2.0,
    image_dimension=(240, 320),
    linear_width=256
)

# Initialize
planner.initialize(None)

# Create environment state with RGB image
rgb_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
env_state = EnvState(rgb_image=rgb_image, depth=np.zeros((240, 320)))

# Generate trajectory
action = planner.compute_planner_trajectory(env_state)
print(f"Generated {len(action.trajectory.waypoints)} waypoints")
```

### Training Example

```python
# Prepare training data
rgb_images = [...]  # List of RGB images
target_actions = [...]  # List of (steering, acceleration) tuples

# Train the model
training_stats = planner.train_episode(rgb_images, target_actions)
print(f"Training loss: {training_stats['average_loss']}")

# Save trained model
planner.save_model("trained_model.pth")
```

### Complete Integration Example

```python
# Run complete simulation loop
def run_simulation(planner, simulator, environment, evaluator, n_steps):
    planner.initialize(None)
    trajectory_history = []
    
    for step in range(n_steps):
        # Get state and sensor data
        current_state = simulator.get_state()
        sensor_output = environment.get_sensor_input(...)
        
        # Plan and execute
        action = planner.compute_trajectory(sensor_output)
        trajectory_history.append(action)
        simulator.do_action(action)
        
        # Evaluate
        score = evaluator.compute_cumulative_score(trajectory_history, current_state)
        print(f"Step {step}: Score = {score}")
```

## Configuration Parameters

### Neural Network Parameters

- `hidden_dimensions`: List of CNN channel sizes (e.g., [32, 64, 128])
- `image_dimension`: Input image size as (height, width)
- `linear_width`: Width of fully connected layers
- `lr`: Learning rate
- `optimizer_type`: "Adam", "SGD", or "AdamW"
- `criterion`: Loss function (default: MSELoss)

### Vehicle Dynamics Parameters

- `horizon_seconds`: Planning horizon (default: 5.0 seconds)
- `sampling_time`: Trajectory sampling interval (default: 0.1 seconds)
- `base_velocity`: Default vehicle velocity (default: 5.0 m/s)
- `max_steering_angle`: Maximum steering angle in radians (default: π/6)
- `max_acceleration`: Maximum acceleration in m/s² (default: 3.0)
- `wheelbase`: Vehicle wheelbase for bicycle model (default: 2.7 meters)

### Training Parameters

- `batch_size`: Training batch size (default: 1)
- `num_epochs`: Training epochs per episode (default: 1)

## Model Management

### Saving and Loading

```python
# Save model with configuration and training history
planner.save_model("my_model.pth")

# Load model
planner.load_model("my_model.pth")

# Create new instance from checkpoint
loaded_planner = ReinforcementLearningPlanner.from_checkpoint("my_model.pth")
```

### Training Statistics

```python
# Get training statistics
stats = planner.get_training_stats()
print(f"Training steps: {stats['num_training_steps']}")
print(f"Average loss: {stats['average_loss']}")
print(f"Latest evaluation: {stats['latest_evaluation']}")

# Reset training history
planner.reset_training_history()
```

## Running the Demo

### Quick Test

```bash
cd monarch
python example_rl_integration.py
```

This will:
1. Create and train an RL planner with synthetic data
2. Run a simulation with the trained planner
3. Demonstrate model saving/loading
4. Show training and evaluation statistics

### Full System Test

```bash
cd monarch
python main.py
```

This will:
1. Test both oscillating and RL planners
2. Compare their performance
3. Use real components if available, otherwise fall back to mocks
4. Demonstrate complete integration

## Advanced Features

### Custom Observation Types

```python
class RGBObservation(Observation):
    """Custom observation type for RL planner"""
    pass

# The planner expects this observation type
assert planner.observation_type() == RGBObservation
```

### Bicycle Model Dynamics

The planner uses realistic vehicle dynamics:

```python
# Bicycle model parameters
turn_radius = wheelbase / tan(steering_angle)
angular_velocity = velocity / turn_radius

# Update vehicle state
heading += angular_velocity * sampling_time
x += velocity * cos(heading) * sampling_time
y += velocity * sin(heading) * sampling_time
```

### Error Handling

The planner includes robust error handling:

- Graceful fallback when no RGB image is available
- Input validation and tensor conversion
- Safe default actions when prediction fails
- Device compatibility (CPU/GPU)

## Performance Considerations

### Inference Speed

- Efficient CNN architecture for real-time performance
- GPU acceleration when available
- Optimized tensor operations

### Memory Usage

- Configurable batch sizes
- Gradient clipping for stability
- Optional training history management

### Training Efficiency

- Depthwise separable convolutions reduce parameters
- Adaptive pooling for variable input sizes
- Multiple optimizer options

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image dimensions
2. **Import errors**: Ensure all dependencies are installed
3. **Training divergence**: Reduce learning rate or add regularization
4. **Poor trajectory quality**: Check vehicle parameter configuration

### Debug Mode

```python
# Enable verbose output
planner = ReinforcementLearningPlanner(...)
planner.initialize(None)

# Check model architecture
print(f"Model parameters: {sum(p.numel() for p in planner.model.parameters())}")
print(f"Device: {planner.device}")
```

## Extension Points

The RL planner can be extended in several ways:

1. **Different CNN architectures**: Modify `_build_model()` method
2. **Alternative vehicle models**: Update `_generate_trajectory_bicycle_model()`
3. **Custom loss functions**: Change the `criterion` parameter
4. **Multi-modal inputs**: Extend to handle additional sensor data
5. **Reinforcement learning**: Add policy gradient or actor-critic methods

## Dependencies

Required packages:
- `torch` (PyTorch)
- `numpy`
- Standard Python libraries (`math`, `time`, `typing`)

Optional packages for full functionality:
- `nuplan-devkit` (for real simulator)
- `omegaconf` (for configuration)
- Additional visualization packages

## License and Contributing

This RL planner implementation is part of the autonomous vehicle simulation framework. Please follow the project's contribution guidelines when making modifications. 