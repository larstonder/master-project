"""
This file contains the definition of the Agent class, which is responsible for
interacting with the simulation environment. The Agent class is designed to be
used in conjunction with the Simulator and EnvironmentModel classes to
perform actions based on sensor data from the simulation.
"""
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox

class Agent:
    """Base class for the agent."""
    def __init__(self):
        pass

    def get_action(self, sensor_output):
        """Get the action based on the sensor output."""
        raise NotImplementedError("This method should be overridden in subclasses.")

class RandomAgent(Agent):
    """Random agent that selects random actions."""
    def __init__(self):
        super().__init__()

    def get_action(self, sensor_output):
        """Select a random trajectory"""
        trajectory = []
        for i in range(10):
            time_point = i * 0.1
            x = i * 0.1
            y = i * 0.1
            heading = 0.0
            velocity = 1.0
            center = StateSE2(x, y, heading)
            oriented_box = OrientedBox(center, 1.0, 1.0, 1.0)
            trajectory.append((time_point, oriented_box, velocity))
        return trajectory
