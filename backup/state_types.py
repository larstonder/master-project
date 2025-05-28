"""
Types for the project
"""

import torch.nn as nn
import numpy as np

# ---------------------------------------SIMULATOR---------------------------------------
class VehicleState:
    """
    Represents the position heading and id of a vehicle in 3D space.
    Attributes:
        x (float): The x-coordinate of the vehicle's position.
        y (float): The y-coordinate of the vehicle's position.
        z (float): The z-coordinate of the vehicle's position.
        heading (float): The heading of the vehicle in radians.
        id (int): The id of the vehicle.
    """

    def __init__(self, x: float, y: float, z: float, heading: float, id: int = None):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        self.id = id

    def __eq__(self, other):
        if not isinstance(other, VehicleState):
            return NotImplemented
        return (
            self.x == other.x
            and self.y == other.y
            and self.z == other.z
            and self.heading == other.heading
        )

    def __repr__(self):
        return f"VehicleState(x={self.x}, y={self.y}, z={self.z}, heading={self.heading})"

class Rotation: # Placeholder as of now
    """
    Represents the rotation of a vehicle in 3D space.
    Attributes:
        roll (float): The roll angle of the vehicle in radians.
        pitch (float): The pitch angle of the vehicle in radians.
        yaw (float): The yaw angle of the vehicle in radians.
    """

    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __eq__(self, other):
        if not isinstance(other, Rotation):
            return NotImplemented
        return (
            self.roll == other.roll
            and self.pitch == other.pitch
            and self.yaw == other.yaw
        )

    def __repr__(self):
        return f"Rotation(roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"


# ---------------------------------------ENVIRONMENT---------------------------------------
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)

class State:
    """
    Represents the state of the simulation at a given time step.
    Attributes:
        ego_pos (VehicleState): The position of the ego vehicle.
        vehicle_pos_list (List[VehicleState]): A list of positions of other vehicles.
        timestamp (float): The timestamp of the state.
    """

    def __init__(
        self,
        ego_pos: VehicleState,
        vehicle_pos_list: list[VehicleState],
        timestamp: float,
        image_idx: int = 0,
    ):
        self.ego_pos = ego_pos
        self.vehicle_pos_list = vehicle_pos_list
        self.timestamp = timestamp
        self.image_idx = image_idx

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.ego_pos == other.ego_pos
            and self.vehicle_pos_list == other.vehicle_pos_list
            and self.timestamp == other.timestamp
            and self.image_idx == other.image_idx
        )

    def __repr__(self):
        return f"State(ego_pos={self.ego_pos}, vehicle_pos_list={self.vehicle_pos_list}, timestamp={self.timestamp}, image_idx={self.image_idx})"

class EnvState:
    def __init__(self, rgb_image, depth):
        self.rgb_image = rgb_image
        self.depth = depth
    
    def __eq__(self, other):
        if not isinstance(other, EnvState):
            return NotImplemented
        return (
            self.rgb_image == other.rgb_image
            and self.depth == other.depth
        )
        
    def __repr__(self):
        return f"EnvState(rgb_image={self.rgb_image}, depth={self.depth})"

# ---------------------------------------AGENT---------------------------------------
class Action:
    """
    Represents an action taken by the agent.
    Attributes:
        steering_angle: [rad]
        accelerate: [m/s^2]
            NOTE: Acceleration is a StateVector2D object with
            acceleration in both x and y direction:
    """
    def __init__(self, turn: float, acceleration: float, brake: float):
        self.steering_angle = steering_angle
        self.acceleration = acceleration

    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.steering_angle == other.steering_angle
            and self.acceleration == other.acceleration
        )

    def __repr__(self):
        return f"Action(steering_angle={self.steering_angle}, acceleration={self.acceleration})"