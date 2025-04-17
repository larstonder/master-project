"""
This module defines the data structures used in the simulation.
It includes classes for representing the state of the simulation,
including the position of the ego vehicle and other vehicles in the environment.
"""

from typing import List

class Position:
    """
    Represents the position and heading of a vehicle in 3D space.
    Attributes:
        x (float): The x-coordinate of the vehicle's position.
        y (float): The y-coordinate of the vehicle's position.
        z (float): The z-coordinate of the vehicle's position.
        heading (float): The heading of the vehicle in radians.
    """
    def __init__(self, x: float, y: float, z: float, heading: float):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading

    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x == other.x and
                self.y == other.y and
                self.z == other.z and
                self.heading == other.heading)

    def __repr__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z}, heading={self.heading})"

class State:
    """
    Represents the state of the simulation at a given time step.
    Attributes:
        ego_pos (Position): The position of the ego vehicle.
        vehicle_pos_list (List[Position]): A list of positions of other vehicles.
        timestamp (float): The timestamp of the state.
    """
    def __init__(self, ego_pos: Position, vehicle_pos_list: List[Position], timestamp: float, image_idx: int = 0):
        self.ego_pos = ego_pos
        self.vehicle_pos_list = vehicle_pos_list
        self.timestamp = timestamp
        self.image_idx = image_idx

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (self.ego_pos == other.ego_pos and
                self.vehicle_pos_list == other.vehicle_pos_list and
                self.timestamp == other.timestamp and
                self.image_idx == other.image_idx)

    def __repr__(self):
        return f"State(ego_pos={self.ego_pos}, vehicle_pos_list={self.vehicle_pos_list}, timestamp={self.timestamp}, image_idx={self.image_idx})"
