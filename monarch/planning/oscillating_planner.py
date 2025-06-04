import os
import math
import numpy as np
from typing import Type, List
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.vehicle_parameters import (
    get_pacifica_parameters,
    VehicleParameters,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Sensors,
    Observation,
)  # Use Sensors for images and DetectionTracks for agents that does not use its observations
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
    KinematicBicycleModel,
)
from monarch.typings.trajectory import Trajectory, Waypoint
from monarch.planning.planner import Planner
from monarch.typings.state_types import VehicleParameters, VehicleState, SystemState, EnvState

NUPLAN_MAP_VERSION = os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")


class OscillatingPlanner(Planner):
    """
    Oscillates from starside to portside.
    Once max steering_angle is reached, the vehicle starts turning in the opposite direction creating a sine-wave trajectory.
    The purpose of the planner is to test whether the simulator handles turning effectively
    The Planner does not take into account the observations it receives
    """

    def __init__(
        self,
        horizon_seconds: int,
        sampling_time: float,
        max_steering_angle: float,
        steering_angle_increment: float,
    ):
        """
        Constructor for OscillatingPlanner
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        :param max_steering_angle: [rad] max_ego_steering angle.
        NOTE:
        - Might update acceleration to apply steering even with constant velocity (accelerates in a given direction)
        - Might be able to remove self.vehicle param
        -- Did not find another motion model so the one in simple_planner is used.
        """
        super().__init__(self.__class__.__name__),  # Add Observation here
        # Init variables from params
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_steering_angle = max_steering_angle
        self.steering_angle_increment = steering_angle_increment

        # Init variables not part of params
        # self.steering_angle = 1.0  # Added steering angle. Don't know if it used yet
        self.max_velocity = 20.0  # Just in case the car accelerates out of control
        self.acceleration = np.array([0.0, 0.0])  # Default acceleration [x, y]
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass"""
        return DetectionsTracks

    def compute_planner_trajectory(
        self, env_state: EnvState, state_history: List[SystemState]
    ) -> Trajectory:
        """
        Inherited, see superclass
        """
        target_speed = 2.0  # Target speed instead of static velocity

        current_time_point = state_history[-1].timestamp
        current_ego = state_history[-1].ego_pos
        previous_ego = state_history[-2 if len(state_history) > 1 else -1].ego_pos

        trajectory_waypoints = [
            Waypoint(
                current_ego.x,
                current_ego.y,
                current_ego.heading,
                current_ego.vehicle_parameters.vx,
                current_ego.vehicle_parameters.vy,
                current_time_point,
            )
        ]

        current_x, current_y, current_heading = (
            current_ego.x,
            current_ego.y,
            current_ego.heading,
        )
        current_vx, current_vy = (
            current_ego.vehicle_parameters.vx,
            current_ego.vehicle_parameters.vy,
        )
        current_steering_angle = current_ego.vehicle_parameters.steering_angle
        previous_steering_angle = previous_ego.vehicle_parameters.steering_angle

        num_steps = int(self.horizon_seconds.time_us / self.sampling_time.time_us)
        dt = self.sampling_time.time_us / 1e6

        for i in range(num_steps):
            new_steering_angle = self._get_new_steering_angle(
                current_steering_angle, previous_steering_angle
            )

            # Update dynamics using bicycle model
            current_speed = math.sqrt(current_vx**2 + current_vy**2)
            if current_speed < target_speed:
                # Simple speed controller
                current_speed = min(current_speed + 1.0 * dt, target_speed)

            # Update heading and position
            angular_velocity = (
                current_speed * math.tan(new_steering_angle)
            ) / get_pacifica_parameters().wheel_base
            current_heading += angular_velocity * dt

            current_vx = current_speed * math.cos(current_heading)
            current_vy = current_speed * math.sin(current_heading)

            current_x += current_vx * dt
            current_y += current_vy * dt

            current_time_point = current_time_point + (self.sampling_time * 1e6)

            trajectory_waypoints.append(
                Waypoint(
                    current_x,
                    current_y,
                    current_heading,
                    current_vx,
                    current_vy,
                    angular_velocity,
                    current_time_point,
                )
            )

            previous_steering_angle = current_steering_angle
            current_steering_angle = new_steering_angle

        return Trajectory(trajectory_waypoints)

    def _get_new_steering_angle(
        self, current_steering_angle: float, previous_steering_angle: float
    ):
        """
        Helper function for computing new steering angle
        :param current_steering_angle: [rad] angle of ego vehicle at current state.
        :param previous_steering_angle: [rad] angle of ego vehicle at previous state.
        :return angle: [rad] Incremented angle
        """
        angle = current_steering_angle
        prev_angle = previous_steering_angle
        # Handle edge cases:
        if angle >= self.max_steering_angle:
            angle = angle - self.steering_angle_increment
        elif abs(angle) >= self.max_steering_angle:
            angle = angle + self.steering_angle_increment

        # Handle default case
        else:
            angle = (
                angle + self.steering_angle_increment
                if angle > prev_angle
                else angle - self.steering_angle_increment
            )

        return angle
