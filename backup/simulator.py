"""
This script initializes a NuPlan simulator and
provides methods to get the current state of the simulation
and perform actions based on a given trajectory.
It uses the NuPlan database and maps to create a simulation environment.
It also includes a function to create a waypoint from a given point.
"""

import os
import traceback
from omegaconf import OmegaConf
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.ego_state import CarFootprint
from state_types import State, VehicleState, Action
from interfaces import Simulator


NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '/data/sets/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '/data/sets/nuplan/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '/data/sets/nuplan/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
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

    def __init__(self):
        super().__init__()
        print("Initializing NuPlan simulator...")
        scenario_builder = NuPlanScenarioBuilder(
            data_root=NUPLAN_DATA_ROOT,
            map_root=NUPLAN_MAPS_ROOT,
            sensor_root=NUPLAN_SENSOR_ROOT,
            db_files=[DB_FILE],
            map_version=NUPLAN_MAP_VERSION,
            vehicle_parameters=get_pacifica_parameters(),
            include_cameras=False,
            verbose=True
        )

        scenario_filter = ScenarioFilter(
            log_names = ["2021.05.12.22.28.35_veh-35_00620_01164"],
            scenario_types = None,
            scenario_tokens = None,
            map_names = None,
            num_scenarios_per_type = None,
            limit_total_scenarios = None,
            timestamp_threshold_s = None,
            ego_displacement_minimum_m = None,
            expand_scenarios = False,
            remove_invalid_goals = False,
            shuffle = False
        )

        worker_config = OmegaConf.create({
            'worker': {
                '_target_': 'nuplan.planning.utils.multithreading.worker_sequential.Sequential',
            }
        })

        worker = build_worker(worker_config)
        self.scenario = scenario_builder.get_scenarios(scenario_filter, worker)[0]

        time_controller = StepSimulationTimeController(self.scenario)
        
        for i in range(1000):
            time_controller.next_iteration()
        
        observation = TracksObservation(self.scenario)
        controller = PerfectTrackingController(self.scenario)

        simulation_setup = SimulationSetup(
            time_controller=time_controller,
            observations=observation,
            ego_controller=controller,
            scenario=self.scenario
        )

        self.simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=None,
            simulation_history_buffer_duration=2.0
        )

        self.simulation.initialize()

        planner_input = self.simulation.get_planner_input()
        history: SimulationHistoryBuffer = planner_input.history
        self.original_ego_state, self.original_observation_state = history.current_state
        self.ego_vehicle_oriented_box = self.original_ego_state.waypoint.oriented_box

        print("NuPlan init completed")


    def get_planner_input(self):
        # print("THIS IS THE PLANNER INPUT", self.simulation.get_planner_input())
        return self.simulation.get_planner_input()

    # !!! REWORK
    def get_state(self) -> State:
        planner_input = self.simulation.get_planner_input()
        history = planner_input.history
        ego_state, observation_state = history.current_state
        
        ego_pos: VehicleState = VehicleState(
            x=ego_state.waypoint.center.x,
            y=ego_state.waypoint.center.y,
            z=606.740,  # height of camera
            heading=ego_state.waypoint.heading
        )
        agent_pos_list: list[VehicleState] = [
            VehicleState(
                x=agent.center.x,
                y=agent.center.y,
                z=606.740,
                heading=agent.center.heading,
                id=agent.metadata.track_token
            )
            for agent in observation_state.tracked_objects.get_agents()
        ]
        state = State(
            ego_pos=ego_pos,
            vehicle_pos_list=agent_pos_list,
            timestamp=ego_state.waypoint.time_point
        )
        return state


    def do_action(self, trajectory):
        # interpolated_trajectory = self.create_interpolated_trajectory(trajectory)
        self.simulation.propagate(trajectory)

    # !!! REWORK
    def create_interpolated_trajectory(self, trajectory: InterpolatedTrajectory):
        """
        Create an interpolated trajectory from a given trajectory.
        :param trajectory: The trajectory to create the interpolated trajectory from.
        :return: The created interpolated trajectory.
        """
        # Convert Waypoints to EgoStates
        ego_states = []
        vehicle_parameters = get_pacifica_parameters()

        for ego_state in trajectory:
            # !!! REDO THIS LOOP?
            waypoint: Waypoint = ego_state.waypoint
            # Extract data from the waypoint
            time_point = waypoint.time_point
            oriented_box = waypoint.oriented_box
            velocity = waypoint.velocity

            # Create a dynamic car state with speed from velocity
            speed = (velocity.x**2 + velocity.y**2)**0.5  # Calculate speed from velocity components

            car_footprint = CarFootprint(
                center=oriented_box.center,
                vehicle_parameters=vehicle_parameters,
            )

            dynamic_car_state = DynamicCarState(
                rear_axle_to_center_dist=vehicle_parameters.cog_position_from_rear_axle,
                rear_axle_velocity_2d=StateVector2D(velocity.x, velocity.y),
                rear_axle_acceleration_2d=StateVector2D(0, 0)  # Assuming no acceleration for simplicity
            )

            # Create an EgoState
            ego_state = EgoState(
                time_point=time_point,
                car_footprint=car_footprint,
                dynamic_car_state=dynamic_car_state,
                tire_steering_angle=0.0,  # Assuming no steering angle for simplicity
                is_in_auto_mode=True,
            )

            ego_states.append(ego_state)

        # Create an interpolated trajectory with EgoState objects
        return InterpolatedTrajectory(ego_states)

    def sandbox_eval(self):
        """
        This function is for testing evaluation of states in the NuPlan simulator.
        """