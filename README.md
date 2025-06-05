# MoNArch Framework

**MoNArch** (Modular NVS Architecture for Closed-loop Simulation) is a modular Python framework for autonomous vehicle simulation, planning, and evaluation. The framework provides a unified interface for developing and testing autonomous driving algorithms with support for multiple simulation backends and evaluation metrics.

## Overview

MoNArCh is designed to bridge the gap between autonomous vehicle research and practical implementation by providing:

- **Modular Architecture**: Clean separation between planning, simulation, rendering, and evaluation components
- **Multiple Planning Algorithms**: Support for simple planners, reinforcement learning agents, and oscillating test planners
- **Simulation Backend Integration**: Built-in support for NuPlan simulation framework
- **Flexible Evaluation**: Comprehensive metrics and evaluation tools for performance assessment
- **Type Safety**: Strong typing system for trajectories, states, and system components

## Architecture

The framework consists of five main components:

### 🧭 Planning (`monarch/planning/`)
- **Abstract Planner Interface**: Base class for all planning algorithms
- **Simple Planner**: Basic straight-line trajectory planner
- **RL Planner**: Reinforcement learning-based planner with CNN architecture
- **Oscillating Planner**: Test planner for validation (sine-wave trajectories)

### 🌍 Simulation (`monarch/simulator/`)
- **Abstract Simulator Interface**: Generic simulation backend interface  
- **NuPlan Integration**: Example integration with the NuPlan autonomous driving dataset and simulator
- **State Management**: Real-time system state tracking and updates

### 🎥 Rendering (`monarch/rendering/`)
- **Abstract Renderer Interface**: Sensor data processing interface
- **OmniRe Integration**: Example integration with the OmniRe rendering engine
- **Multi-modal Sensors**: RGB cameras, depth sensors, and environmental data

### 📊 Evaluation (`monarch/evaluation/`)
- **Multiple Evaluators**: Simple, RL-based, and comprehensive evaluation metrics
- **Sandbox Environment**: Isolated testing environment for algorithm validation
- **Performance Metrics**: Comprehensive scoring and analysis tools

### 🔧 Utilities (`monarch/utils/`, `monarch/typings/`)
- **Image Processing**: Utilities for sensor data handling
- **Path Operations**: Trajectory and path manipulation tools
- **Type Definitions**: Strong typing for states, trajectories, and system components

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (for RL planner)
- NuPlan dataset (for simulation)

### Dependencies
```bash
# Core dependencies
pip install numpy torch torchvision
pip install omegaconf dataclasses-json

# Optional dependencies
pip install matplotlib
pip install opencv-python
```

<!-- ### Environment Setup
Set the following environment variables for NuPlan integration:

```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/data"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/maps"  
export NUPLAN_DB_FILES="/path/to/nuplan/splits"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
``` -->

## Quick Start

### Basic Simulation Loop
```python
from monarch.simulator.abstract_simulator import AbstractSimulator
from monarch.rendering.abstract_renderer import AbstractRenderer  
from monarch.planning.abstract_planner import AbstractPlanner

def run_simulation(
    n_steps: int,
    simulator: AbstractSimulator,
    renderer: AbstractRenderer,
    planner: AbstractPlanner
):
    """
    Main simulation loop following the MoNArCh architecture:
    Simulator -> Renderer -> Planner -> Simulator
    """
    
    # Initialize simulation state
    original_state = simulator.get_state()
    current_state = original_state
    state_history = [original_state]
    
    for step in range(n_steps):
        # Step 1: Get current simulation state
        last_state = current_state
        current_state = simulator.get_state()
        state_history.append(current_state)
        
        # Step 2: Renderer processes state to synthesized sensor input
        sensor_input = renderer.get_sensor_input(
            original_state, last_state, current_state
        )
        
        # Step 3: Planner computes trajectory from sensor input
        trajectory = planner.compute_planner_trajectory(
            sensor_input, state_history
        )
        
        # Step 4: Execute trajectory in simulator
        simulator.do_action(trajectory)
    
    return state_history

# Example usage with concrete implementations
# Note: Specific simulators (NuPlan) and renderers (OmniRe) are available
simulator = YourSimulatorImplementation()
renderer = YourRendererImplementation()  
planner = YourPlannerImplementation()

# Run the simulation
history = run_simulation(
    n_steps=100,
    simulator=simulator,
    renderer=renderer, 
    planner=planner
)
```

<!-- ### Simple Planning Example
```python
from monarch.planning.simple_planner import SimplePlanner

# Initialize a basic straight-line planner
planner = SimplePlanner(
    horizon_seconds=5.0,
    sampling_time=0.1,
    acceleration=[2.0, 0.0],
    max_velocity=15.0,
    steering_angle=0.0
)
```

### Reinforcement Learning Planner
```python
from monarch.planning.rl_planner import ReinforcementLearningPlanner

# Initialize RL planner with CNN architecture
rl_planner = ReinforcementLearningPlanner(
    hidden_dimensions=[64, 128, 256],
    image_dimension=[224, 224],
    horizon_seconds=3.0,
    sampling_time=0.1,
    max_steering_angle=0.6,
    batch_size=32,
    lr=0.001
)

# Train the model (requires training data)
rl_planner.train(training_data, num_epochs=100)
``` -->

<!-- ## Core Data Types

### Trajectory and Waypoints
```python
from monarch.typings.trajectory import Trajectory, Waypoint

# Create waypoints
waypoints = [
    Waypoint(x=0.0, y=0.0, heading=0.0, vx=10.0, vy=0.0, timestamp=0.0),
    Waypoint(x=10.0, y=0.0, heading=0.0, vx=10.0, vy=0.0, timestamp=1.0)
]

# Create trajectory
trajectory = Trajectory(waypoints)
```

### System State
```python
from monarch.typings.state_types import SystemState, VehicleState

# Define ego vehicle state
ego_state = VehicleState(x=0.0, y=0.0, z=0.0, heading=0.0, id=-1)

# Create system state
system_state = SystemState(
    ego_pos=ego_state,
    vehicle_pos_list=[],  # Other vehicles
    timestamp=0.0
)
``` -->

<!-- ## Advanced Usage

### Custom Planner Implementation
```python
from monarch.planning.abstract_planner import AbstractPlanner

class CustomPlanner(AbstractPlanner):
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "CustomPlanner"
    
    def compute_planner_trajectory(self, env_state: EnvState, state_history: List[SystemState]) -> Trajectory:
        # Implement your planning logic here
        return Trajectory(waypoints)
```

### Custom Evaluator
```python
from monarch.evaluation.abstract_evaluator import AbstractEvaluator

class CustomEvaluator(AbstractEvaluator):
    @property
    def name(self) -> str:
        return "CustomEvaluator"
    
    def compute_cumulative_score(self, history, scenario) -> float:
        # Implement your evaluation metrics
        return self._calculate_score(history, scenario)
``` -->

## Performance Features

- **GPU Acceleration**: RL planner supports CUDA for faster training and inference
- **Vectorized Operations**: NumPy-based computations for efficient trajectory processing
- **Modular Design**: Easy to extend and customize individual components
- **Type Safety**: Comprehensive type hints for better development experience

<!-- ## Integration with NuPlan

MoNArCh provides seamless integration with the NuPlan ecosystem:

- **Scenario Loading**: Direct access to NuPlan scenarios and maps
- **Sensor Data**: RGB cameras, depth sensors, and environmental observations
- **Vehicle Dynamics**: Realistic vehicle models and physics simulation
- **Evaluation Metrics**: Compatible with NuPlan's evaluation framework -->

<!-- ## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Implement your changes with proper type hints
4. Add tests for new functionality
5. Submit a pull request -->

<!-- ### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all public methods
- Include docstrings for classes and methods
- Write unit tests for new components -->

<!-- ## Testing

```bash
# Run all tests
python -m pytest

# Run specific component tests
python -m pytest evaluation/test_evaluator.py
``` -->

## License

[Add your license information here]

## Citation

If you use MoNArCh in your research, please cite:

```bibtex
@software{monarch_framework,
  title={MoNArCh: Modular NVS Architecture for Closed-loop Simulation},
  author={Tønder Lars and Reinseth, Johannes},
  year={2024},
  url={https://github.com/lars-tønder/master-project}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in each module
- Review the example implementations in the codebase

---

**MoNArCh** - Advancing autonomous vehicle research through modular, extensible simulation and planning frameworks. 