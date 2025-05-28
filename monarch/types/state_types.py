from dataclasses import dataclass
from typing import Iterable
import numpy as np
import numpy.typing as npt

@dataclass
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

@dataclass
class Point2D:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location
    __slots__ = "x", "y"

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y)
        """
        return iter((self.x, self.y))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        return np.array([self.x, self.y], dtype=np.float64)

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


@dataclass
class State(Point2D):
    """
    State - representing [x, y, heading]
    """

    heading: float  # [rad] heading of a state
    __slots__ = "heading"

    @property
    def point(self) -> Point2D:
        """
        Gets a point from the State
        :return: Point with x and y from State
        """
        return Point2D(self.x, self.y)

    def as_matrix(self) -> npt.NDArray[np.float32]:
        """
        :return: 3x3 2D transformation matrix representing the State.
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), self.x],
                [np.sin(self.heading), np.cos(self.heading), self.y],
                [0.0, 0.0, 1.0],
            ]
        )

    def as_matrix_3d(self) -> npt.NDArray[np.float32]:
        """
        :return: 4x4 3D transformation matrix representing the State projected to 3D.
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), 0.0, self.x],
                [np.sin(self.heading), np.cos(self.heading), 0.0, self.y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

@dataclass
class SystemState:
    """
    Represents the state of the system at a given time step.
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

@dataclass
class EnvState:
    """
    Represents the state of the environment at a given time step.
    Attributes:
        rgb_image (np.ndarray): The RGB image of the environment.
        depth (np.ndarray): The depth image of the environment.
    """
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
