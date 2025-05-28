import math
import numpy as np
from typing import Type
from ..types.state_types import EnvState, VehicleState, SystemState
from ..types.action import Action, Trajectory, Waypoint
from ..types.observation_type import Observation
from .planner import Planner


class OscillatingObservation(Observation):
    """
    Simple observation type for the oscillating planner.
    """
    pass


class OscillatingPlanner(Planner):
    """
    Oscillates from starboard to portside.
    Once max steering_angle is reached, the vehicle starts turning in the opposite direction creating a sine-wave trajectory.
    The purpose of the planner is to test whether the simulator handles turning effectively.
    The Planner creates oscillating behavior based on the current timestamp.
    """

    def __init__(
        self,
        horizon_seconds: float,
        sampling_time: float,
        max_steering_angle: float,
        steering_angle_increment: float,
        base_velocity: float = 5.0
    ):
        """
        Constructor for OscillatingPlanner
        :param horizon_seconds: [s] time horizon being planned.
        :param sampling_time: [s] sampling timestep.
        :param max_steering_angle: [rad] maximum steering angle.
        :param steering_angle_increment: [rad] increment step for steering angle changes.
        :param base_velocity: [m/s] base velocity for the vehicle.
        """
        self.horizon_seconds = horizon_seconds
        self.sampling_time = sampling_time
        self.max_steering_angle = max_steering_angle
        self.steering_angle_increment = steering_angle_increment
        self.base_velocity = base_velocity
        
        # Track the current direction of oscillation
        self._steering_direction = 1.0  # 1 for right, -1 for left
        self._current_steering_angle = 0.0
        
        # Oscillation parameters
        self._oscillation_frequency = 0.5  # Hz - how fast to oscillate
        self._initialized = False

    @property
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self, initialization) -> None:
        """
        Initialize planner
        :param initialization: instance of initialization (not used in this simple planner)
        """
        self._initialized = True

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass"""
        return OscillatingObservation

    def compute_trajectory(self, current_input):
        """
        Computes the ego vehicle trajectory
        :param current_input: Current input containing the system state
        :return: Action containing the planned trajectory
        """
        return self.compute_planner_trajectory(current_input)

    def compute_planner_trajectory(self, current_input) -> Action:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Current system state or input
        :return: Action containing the planned trajectory
        """
        # Extract current state information
        if hasattr(current_input, 'ego_pos'):
            # If current_input is a SystemState
            current_pos = current_input.ego_pos
            timestamp = current_input.timestamp
        elif hasattr(current_input, 'timestamp'):
            # If current_input has timestamp but different structure
            timestamp = current_input.timestamp
            current_pos = VehicleState(0.0, 0.0, 0.0, 0.0, 0)  # Default if not available
        else:
            # Use current time if no timestamp available
            import time
            timestamp = time.time()
            current_pos = VehicleState(0.0, 0.0, 0.0, 0.0, 0)
        
        # Generate waypoints for the trajectory
        waypoints = []
        
        # Calculate number of steps
        num_steps = int(self.horizon_seconds / self.sampling_time)
        
        # Starting position and heading
        current_x = current_pos.x if hasattr(current_pos, 'x') else 0.0
        current_y = current_pos.y if hasattr(current_pos, 'y') else 0.0
        current_heading = current_pos.heading if hasattr(current_pos, 'heading') else 0.0
        
        # Update current steering angle based on oscillation pattern
        self._update_steering_angle(timestamp)
        
        for i in range(num_steps + 1):
            # Calculate time offset
            time_offset = i * self.sampling_time
            future_timestamp = timestamp + time_offset
            
            # Update steering angle for this timestep
            steering_angle = self._calculate_steering_angle_at_time(future_timestamp)
            
            # Update heading based on steering angle
            heading_change = steering_angle * self.sampling_time
            new_heading = current_heading + heading_change
            
            # Calculate velocity components
            velocity_x = self.base_velocity * math.cos(new_heading)
            velocity_y = self.base_velocity * math.sin(new_heading)
            
            # Update position
            new_x = current_x + velocity_x * self.sampling_time
            new_y = current_y + velocity_y * self.sampling_time
            
            # Create waypoint
            waypoint = Waypoint(new_x, new_y, new_heading)
            waypoints.append(waypoint)
            
            # Update current position for next iteration
            current_x, current_y, current_heading = new_x, new_y, new_heading
        
        # Create trajectory and action
        trajectory = Trajectory(waypoints)
        action = Action(trajectory)
        
        return action

    def _update_steering_angle(self, timestamp: float) -> None:
        """
        Update the current steering angle based on timestamp
        :param timestamp: Current timestamp
        """
        self._current_steering_angle = self._calculate_steering_angle_at_time(timestamp)

    def _calculate_steering_angle_at_time(self, timestamp: float) -> float:
        """
        Calculate steering angle at a specific time using oscillation pattern
        :param timestamp: Time at which to calculate steering angle
        :return: Steering angle in radians
        """
        # Create oscillating pattern based on timestamp
        oscillation_phase = 2 * math.pi * self._oscillation_frequency * timestamp
        
        # Generate oscillating steering angle
        # Use sine wave to create smooth oscillation between -max_steering_angle and +max_steering_angle
        normalized_angle = math.sin(oscillation_phase)
        steering_angle = normalized_angle * self.max_steering_angle
        
        return steering_angle

    def _get_new_steering_angle(
        self, current_steering_angle: float, previous_steering_angle: float
    ) -> float:
        """
        Helper function for computing new steering angle using increment-based approach
        :param current_steering_angle: [rad] angle of ego vehicle at current state.
        :param previous_steering_angle: [rad] angle of ego vehicle at previous state.
        :return angle: [rad] Incremented angle
        """
        angle = current_steering_angle
        prev_angle = previous_steering_angle
        
        # Handle edge cases:
        if angle >= self.max_steering_angle:
            angle = angle - self.steering_angle_increment
            self._steering_direction = -1.0
        elif angle <= -self.max_steering_angle:
            angle = angle + self.steering_angle_increment
            self._steering_direction = 1.0
        else:
            # Continue in current direction
            angle += self._steering_direction * self.steering_angle_increment
        
        return angle


if __name__ == "__main__":
    # Example usage
    planner = OscillatingPlanner(
        horizon_seconds=5.0, 
        sampling_time=0.1, 
        max_steering_angle=math.pi/6, 
        steering_angle_increment=0.05
    )
    
    # Initialize planner
    planner.initialize(None)
    
    # Create a simple system state for testing
    ego_pos = VehicleState(0.0, 0.0, 0.0, 0.0, 1)
    system_state = SystemState(ego_pos, [], 0.0)
    
    # Generate trajectory
    action = planner.compute_planner_trajectory(system_state)
    
    print(f"Generated trajectory with {len(action.trajectory.waypoints)} waypoints")
    for i, waypoint in enumerate(action.trajectory.waypoints[:5]):  # Print first 5 waypoints
        print(f"Waypoint {i}: x={waypoint.x:.2f}, y={waypoint.y:.2f}, heading={waypoint.heading:.2f}")
