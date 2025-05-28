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

@dataclass
class Trajectory:
    """
    Represents the trajectory of the system, in the form of a list of waypoints.
    Attributes:
        waypoints (List[Waypoint]): The waypoints of the trajectory.
    """

    def __init__(self, waypoints: List['Waypoint']):
        self.waypoints = waypoints

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return NotImplemented
        return self.waypoints == other.waypoints

@dataclass
class Waypoint:
    """
    Represents a waypoint in the trajectory.
    Attributes:
        x (float): The x-coordinate of the waypoint.
        y (float): The y-coordinate of the waypoint.
        heading (float): The heading of the waypoint.
    """
    def __init__(self, x: float, y: float, heading: float):
        self.x = x
        self.y = y
        self.heading = heading

    def __eq__(self, other):
        if not isinstance(other, Waypoint):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.heading == other.heading