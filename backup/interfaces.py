"""
Interfaces for the project
"""
from state_types import State, Action, EnvState


# ---------------------------------------SIMULATOR---------------------------------------
class Simulator:
    """Base class for the simulator."""
    def __init__(self):
        pass

    def get_state(self):
        """Get the current state of the simulation."""
        raise NotImplementedError("This method should be overridden in subclasses.")

    def do_action(self, action: Action):
        """Perform an action in the simulation."""
        raise NotImplementedError("This method should be overridden in subclasses.")


# ---------------------------------------ENVIRONMENT---------------------------------------
class Environment:
    """Base class for the environment model."""
    def __init__(self):
        pass

    def get_sensor_output(self, state: State):
        """Generate sensor output for the given simulation state."""
        raise NotImplementedError("This method should be overridden by subclasses.")


# ---------------------------------------AGENT---------------------------------------
class Agent:
    """Base class for the agent."""
    def __init__(self):
        pass

    def act(self, sensor_outputs: list[EnvState]):
        """Perform an action based on the sensor output."""
        raise NotImplementedError("This method should be overridden in subclasses.")

    def compute_loss(self):
        """Get the action based on the sensor output."""
        raise NotImplementedError("This method should be overridden in subclasses.")

# ---------------------------------------Planner---------------------------------------
# Planner interface might be redundant
class Planner:
    """Planner interface"""
    def __init__(self):
        pass