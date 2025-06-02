import os
import math
import numpy as np
from typing import Type
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters, VehicleParameters
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Sensors, Observation # Use Sensors for images and DetectionTracks for agents that does not use its observations
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel

NUPLAN_MAP_VERSION = os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

class OscillatingPlanner(AbstractPlanner):
    """
    Oscillates from starside to portside. 
    Once max steering_angle is reached, the vehicle starts turning in the opposite direction creating a sine-wave trajectory.
    The purpose of the planner is to test whether the simulator handles turning effectively
    The Planner does not take into account the observations it receives
    """
    def __init__(self, horizon_seconds: int, sampling_time: float, max_steering_angle: float, steering_angle_increment):
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
        # Init variables from params
        self.horizon_seconds = horizon_seconds
        self.sampling_time = sampling_time
        self.max_velocity = 20.0 # Just in case the car accelerates out of control
        self.max_steering_angle = max_steering_angle
        self.acceleration = np.array([0.0, 0.0])
        self.steering_angle_increment = steering_angle_increment

        # Init variables not part of params
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited, see superclass.
        Not utilized in this planner class as the planner only oscillates
        """
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass"""
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        history = current_input.history
        ego_state, observation = history.ego_states[-1]
        previous_ego_state, observation = history.ego_states[-2]
        print("THIS IS WHAT OBSERVATION IS: ", observation)

        current_state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle( # Might need to change this as the acceleration is different from the SimplePlanner
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                self.acceleration,
            ),
            tire_steering_angle = self.steering_angle,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        previous_state = EgoState(
            car_footprint=previous_ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle( # Might need to change this as the acceleration is different from the SimplePlanner
                previous_ego_state.car_footprint.rear_axle_to_center_dist,
                previous_ego_state.dynamic_car_state.rear_axle_velocity_2d,
                self.acceleration,
            ),
            tire_steering_angle = self.steering_angle,
            is_in_auto_mode=True,
            time_point=previous_ego_state.time_point,
        )

        trajectory = list[EgoState] = [state]
        for _ in range(int(self.horizon_seconds.time_us / self.sampling_time.time_us)):
            # Testing first with tyre steering angle
            steering_angle = current_state.dynamic_car_state.tire_steering_rate 
            if (
                state.tire_steering_angle >= self.max_steering_angle
            ): # Check if steering_angle right is to large
                steering_angle -= self.steering_angle_increment
            elif (
                abs(state.tire_steering_angle) >= self.max_steering_angle
            ):  # Check if steering_angle right is to large
                steering_angle += self.steering_angle_increment
            else:
                # Increment steering angle in the direction currently steering
                if current_state.dynamic_car_state.tire_steering_rate > previous_state.dynamic_car_state.tire_steering_rate: 
                    steering_angle += self.steering_angle_increment
                else:
                    steering_angle -= self.steering_angle_increment

            new_state = EgoState.build_from_rear_axle( # Might need another method if this one does not work
                tire_steering_angle=steering_angle
            )
            state = self.motion_model.propagate_state(new_state, state.dynamic_car_state, self.sampling_time)
            trajectory.append(state)


if __name__ == "__main__":
    planner = OscillatingPlanner(10, 1.0, 5.0, 1.0)
    report = planner.generate_planner_report()
