import os
import math
import numpy as np
from typing import Type
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
from ..types.trajectory import Trajectory, Waypoint
from .planner import Planner

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
        super().__init__(self.__class__.__name__), # Add Observation here 
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
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        static_velocity_magnitude = 2.0

        history = current_input.history
        ego_state = history.ego_states[-1]
        previous_ego_state = history.ego_states[
            -2 if len(history.ego_states) > 1 else -1
        ]

        trajectory: list[EgoState] = [ego_state]
        current_steering_angle = ego_state.tire_steering_angle
        previous_steering_angle = previous_ego_state.tire_steering_angle
        for i in range(int(self.horizon_seconds.time_us / self.sampling_time.time_us)):
            new_steering_angle = self._get_new_steering_angle(
                current_steering_angle, previous_steering_angle
            )

            split_state = ego_state.to_split_state()
            split_state.linear_states[-1] = new_steering_angle
            split_state.linear_states[3] = static_velocity_magnitude * math.cos(
                new_steering_angle
            )
            split_state.linear_states[4] = static_velocity_magnitude * math.sin(
                new_steering_angle
            )
            split_state.linear_states[5] = 0
            split_state.linear_states[6] = 0

            state = EgoState.from_split_state(split_state)
            state = self.motion_model.propagate_state(
                state, state.dynamic_car_state, self.sampling_time
            )
            center = state.waypoint.oriented_box.center
            trajectory.append(Waypoint(center.x, center.y, state.waypoint.heading))

            previous_steering_angle = current_steering_angle
            current_steering_angle = new_steering_angle
        return Trajectory(trajectory)

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
