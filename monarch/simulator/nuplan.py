"""
This script initializes a NuPlan simulator and
provides methods to get the current state of the simulation
and perform actions based on a given trajectory.
It uses the NuPlan database and maps to create a simulation environment.
It also includes a function to create a waypoint from a given point.
"""

import os
from omegaconf import OmegaConf
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.simulation.controller.perfect_tracking import (
    PerfectTrackingController,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.ego_state import CarFootprint
from nuplan.common.actor_state.waypoint import Waypoint as NuPlanWaypoint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint

from monarch.typings.state_types import SystemState, VehicleState, VehicleParameters
from monarch.typings.action import Action
from monarch.simulator.simulator import Simulator
from monarch.typings.trajectory import Trajectory


NUPLAN_DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT", "/data/sets/nuplan")
NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT", "/data/sets/nuplan/maps")
NUPLAN_DB_FILES = os.getenv(
    "NUPLAN_DB_FILES", "/data/sets/nuplan/nuplan-v1.1/splits/mini"
)
NUPLAN_MAP_VERSION = os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs"
DB_FILE = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.05.12.22.28.35_veh-35_00620_01164.db"
MAP_NAME = "us-nv-las-vegas"


class NuPlan(Simulator):
    """
    NuPlan simulator class that initializes the NuPlan simulation environment.
    It uses the NuPlan database and maps to create a simulation environment.
    It provides methods to get the current state of the simulation and perform
    actions based on a given trajectory.
    NOTE: https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md metrics description
    """

    def __init__(self, scenario_name: str, start_timestamp_s: float = 1000.0):
        print("Initializing NuPlan simulator...")
        
        db_file = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/{scenario_name}.db"
        
        scenario_builder = NuPlanScenarioBuilder(
            data_root=NUPLAN_DATA_ROOT,
            map_root=NUPLAN_MAPS_ROOT,
            sensor_root=NUPLAN_SENSOR_ROOT,
            db_files=[db_file],
            map_version=NUPLAN_MAP_VERSION,
            vehicle_parameters=get_pacifica_parameters(),
            include_cameras=False,
            verbose=True,
        )

        scenario_filter = ScenarioFilter(
            log_names=[scenario_name],
            scenario_types=None,
            scenario_tokens=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=None,
            timestamp_threshold_s=start_timestamp_s,
            ego_displacement_minimum_m=None,
            expand_scenarios=False,
            remove_invalid_goals=False,
            shuffle=False,
        )

        worker_config = OmegaConf.create(
            {
                "worker": {
                    "_target_": "nuplan.planning.utils.multithreading.worker_sequential.Sequential",
                }
            }
        )

        worker = build_worker(worker_config)
        self._scenario = scenario_builder.get_scenarios(scenario_filter, worker)[0]

        time_controller = StepSimulationTimeController(self._scenario)
        
        for i in range(1000):
            time_controller.next_iteration()
        
        observation = TracksObservation(self._scenario)
        controller = PerfectTrackingController(self._scenario)

        self.simulation_setup = SimulationSetup(
            time_controller=time_controller,
            observations=observation,
            ego_controller=controller,
            scenario=self._scenario,
        )

        self.reset()

        planner_input = self._simulation.get_planner_input()
        history: SimulationHistoryBuffer = planner_input.history
        self.original_ego_state, self.original_observation_state = history.current_state
        self.ego_vehicle_oriented_box = self.original_ego_state.waypoint.oriented_box
        
        super().__init__(self._scenario, self._simulation)

        print("NuPlan init completed")
    
    def reset(self):
        self._simulation = Simulation(
            simulation_setup=self.simulation_setup,
            callback=None,
            simulation_history_buffer_duration=2.0,
        )
        self._simulation.initialize()

    def get_planner_input(self):
        return self._simulation.get_planner_input()

    def get_state(self) -> SystemState:
        planner_input = self._simulation.get_planner_input()
        history = planner_input.history
        ego_state, observation_state = history.current_state
        
        velocity_2d = ego_state.dynamic_car_state.rear_axle_velocity_2d
        
        vehicle_parameters = VehicleParameters(
            vx = velocity_2d.x,
            vy = velocity_2d.y,
            angular_velocity = ego_state.dynamic_car_state.angular_velocity,
            steering_angle = ego_state.tire_steering_angle,
        )

        ego_pos: VehicleState = VehicleState(
            x=ego_state.waypoint.center.x,
            y=ego_state.waypoint.center.y,
            z=606.740,  # height of camera
            heading=ego_state.waypoint.heading,
            id=-1,
            vehicle_parameters = vehicle_parameters
        )
        
        agent_pos_list: list[VehicleState] = [
            VehicleState(
                x=agent.center.x, y=agent.center.y, z=606.740, heading=agent.center.heading, id=agent.track_token
            )
            for agent in observation_state.tracked_objects.get_agents()
        ]
        state = SystemState(
            ego_pos=ego_pos,
            vehicle_pos_list=agent_pos_list,
            timestamp=ego_state.waypoint.time_point.time_us,
        )
        return state
    
    def _calculate_acceleration(self, first_waypoint: Waypoint, second_waypoint: Waypoint) -> StateVector2D:
        """
        Calculate 2D acceleration from consecutive waypoints.
        """
        first_velocity_2d = StateVector2D(first_waypoint.vx, first_waypoint.vy)
        second_velocity_2d = StateVector2D(second_waypoint.vx, second_waypoint.vy)
        time_diff = second_waypoint.timestamp - first_waypoint.timestamp
        delta_vx = second_waypoint.vx - first_waypoint.vx
        delta_vy = second_waypoint.vy - first_waypoint.vy
        return StateVector2D(delta_vx / time_diff, delta_vy / time_diff)

    def _calculate_angular_acceleration(self, first_waypoint: Waypoint, second_waypoint: Waypoint) -> float:
        """
        Calculate angular acceleration from consecutive waypoints.
        """
        first_angular_velocity = first_waypoint.angular_velocity
        second_angular_velocity = second_waypoint.angular_velocity
        time_diff = second_waypoint.timestamp - first_waypoint.timestamp
        return (second_angular_velocity - first_angular_velocity) / time_diff

    def _create_abstract_trajectory(self, trajectory: Trajectory) -> AbstractTrajectory:
        """
        Convert custom Trajectory to NuPlan's AbstractTrajectory.
        
        :param trajectory: The custom trajectory to convert
        :return: AbstractTrajectory that can be used with NuPlan simulation
        """
        # Convert custom waypoints to NuPlan EgoState objects
        ego_states = []
        
        # Get vehicle parameters for consistent dynamics
        vehicle_params = get_pacifica_parameters()
        
        for i, waypoint in enumerate(trajectory.waypoints):
            # Create time point from timestamp (convert to microseconds)
            time_point = TimePoint(int(waypoint.timestamp))
            
            # Create pose (position + heading) - assume this is the center pose
            center_pose = StateSE2(x=waypoint.x, y=waypoint.y, heading=waypoint.heading)
            
            # Create velocity vector
            center_velocity_2d = StateVector2D(waypoint.vx, waypoint.vy)
            
            # Create acceleration (assume zero for simplicity)
            if i < len(trajectory.waypoints) - 1:
                center_acceleration_2d = self._calculate_acceleration(waypoint, trajectory.waypoints[i+1])
                angular_acceleration = self._calculate_angular_acceleration(waypoint, trajectory.waypoints[i+1])
            else:
                center_acceleration_2d = StateVector2D(0.0, 0.0)
                angular_acceleration = 0.0
            
            # Create EgoState using the build_from_center method
            ego_state = EgoState.build_from_center(
                center=center_pose,
                center_velocity_2d=center_velocity_2d,
                center_acceleration_2d=center_acceleration_2d,
                tire_steering_angle=0.0,  # Could be improved by tracking steering
                time_point=time_point,
                vehicle_parameters=vehicle_params,
                is_in_auto_mode=True,
                angular_vel=waypoint.angular_velocity,  # Could be improved by calculating from consecutive waypoints
                angular_accel=angular_acceleration
            )
            
            ego_states.append(ego_state)
        
        # Create and return interpolated trajectory with EgoState objects
        return InterpolatedTrajectory(ego_states)

    def do_action(self, trajectory: Trajectory):
        abstract_trajectory = self._create_abstract_trajectory(trajectory)
        self._simulation.propagate(abstract_trajectory)
