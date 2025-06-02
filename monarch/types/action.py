from dataclasses import dataclass
from typing import List

@dataclass
class Action:
    """
    Represents the action of the system at a given time step.
    Attributes:
        trajectory (Trajectory): The trajectory of the system.
    """
    def __init__(self, trajectory: 'Trajectory'):
        self.trajectory = trajectory
    
    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return self.trajectory == other.trajectory