import numpy as np
from monarch.planning.planner import Planner
from monarch.typings.trajectory import Trajectory, Waypoint
from monarch.typings.state_types import SystemState, EnvState, VehicleState, VehicleParameters
import math
from typing import List

class SimplePlanner(Planner):
    """
    Planner going straight.
    """

    def __init__(self, horizon_seconds: float, sampling_time: float, acceleration: np.array, max_velocity: float, steering_angle: float):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param acceleration: [m/s^2] acceleration of the vehicle.
        :param max_velocity: [m/s] max velocity of the vehicle.
        :param steering_angle: [rad] steering angle of the vehicle.
        """
        super().__init__(self.__class__.__name__)
        self.horizon_seconds = horizon_seconds
        self.sampling_time = sampling_time
        self.acceleration = acceleration
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        
    def compute_planner_trajectory(self, env_state: EnvState, state_history: List[SystemState]) -> Trajectory:
        """
        Computes the trajectory of the vehicle driving straight ahead.
        :param env_state: [EnvState] current environment state.
        :param state_history: [List[SystemState]] history of system states.
        :return: [Trajectory] trajectory of the vehicle.
        """
        # Get the current ego state from the latest system state
        current_ego = state_history[-1].ego_pos
        current_time_point = state_history[-1].timestamp
        # Extract current state values
        x, y = current_ego.x, current_ego.y
        heading = current_ego.heading
        
        # Use current velocity or calculate from vx, vy if available
        if hasattr(current_ego, 'vx') and hasattr(current_ego, 'vy'):
            velocity = math.sqrt(current_ego.vx**2 + current_ego.vy**2)
        else:
            # Fallback if velocity fields aren't available
            velocity = 0.0
            
        vx = current_ego.vehicle_parameters.vx
        vy = current_ego.vehicle_parameters.vy
        
        angular_velocity = current_ego.vehicle_parameters.angular_velocity

        # Prepare trajectory starting with current position
        trajectory = [Waypoint(x, y, heading, vx, vy, angular_velocity, current_time_point)]
        time_steps = int(self.horizon_seconds / self.sampling_time)
        dt = self.sampling_time

        for i in range(time_steps):
            # Accelerate up to max_velocity
            velocity = min(velocity + self.acceleration[0] * dt, self.max_velocity)
            
            # Move straight ahead (no steering angle change for simple planner)
            x += velocity * math.cos(heading) * dt
            y += velocity * math.sin(heading) * dt
            
            vx = velocity * math.cos(heading)
            vy = velocity * math.sin(heading)
            
            timepoint = current_time_point + (i+1) * dt * 1e6
            
            # Keep the same heading (straight driving)
            trajectory.append(Waypoint(x, y, heading, vx, vy, angular_velocity, timepoint))

        return Trajectory(trajectory)
        