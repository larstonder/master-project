"""
This script initializes a NuPlan simulator and
provides methods to get the current state of the simulation
and perform actions based on a given trajectory.
It uses the NuPlan database and maps to create a simulation environment.
It also includes a function to create a waypoint from a given point.
"""

import os
from sim_types import State, Position
from omegaconf import OmegaConf
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '/data/sets/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '/data/sets/nuplan/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '/data/sets/nuplan/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs"
DB_FILE = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.05.12.22.28.35_veh-35_00620_01164.db"
MAP_NAME = "us-nv-las-vegas"


class Simulator:
    """Base class for the simulator."""
    def __init__(self):
        pass

    def get_state(self):
        """Get the current state of the simulation."""
        raise NotImplementedError("This method should be overridden in subclasses.")

    def do_action(self, action):
        """Perform an action in the simulation."""
        raise NotImplementedError("This method should be overridden in subclasses.")

class NuPlan(Simulator):
    """
    NuPlan simulator class that initializes the NuPlan simulation environment.
    It uses the NuPlan database and maps to create a simulation environment.
    It provides methods to get the current state of the simulation and perform
    actions based on a given trajectory.
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
        scenario = scenario_builder.get_scenarios(scenario_filter, worker)[0]

        time_controller = StepSimulationTimeController(scenario)
        observation = TracksObservation(scenario)
        controller = PerfectTrackingController(scenario)

        simulation_setup = SimulationSetup(
            time_controller=time_controller,
            observations=observation,
            ego_controller=controller,
            scenario=scenario
        )

        self.simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=None,
            simulation_history_buffer_duration=2.0
        )

        self.simulation.initialize()

        planner_input = self.simulation.get_planner_input()
        history = planner_input.history
        self.original_ego_state, self.original_observation_state = history.current_state
        
        print(self.original_ego_state)

        self.ego_vehicle_oriented_box = self.original_ego_state.waypoint.oriented_box

        print("NuPlan initialized.")

    def get_state(self) -> State:
        planner_input = self.simulation.get_planner_input()
        history = planner_input.history
        ego_state, observation_state = history.current_state

        ego_pos: Position = Position(
            x=ego_state.waypoint.center.x,
            y=ego_state.waypoint.center.y,
            z=0,
            heading=ego_state.waypoint.heading
        )
        agent_pos_list: list[Position] = [
            Position(
                x=agent.center.x,
                y=agent.center.y,
                z=0,
                heading=agent.center.heading
            )
            for agent in observation_state.tracked_objects.get_agents()
        ]
        state = State(
            ego_pos=ego_pos,
            vehicle_pos_list=agent_pos_list,
            timestamp=ego_state.waypoint.time_point
        )
        return state

    def do_action(self, action):
        trajectory = action
        interpolated_trajectory = self.create_interpolated_trajectory(trajectory)
        self.simulation.propagate(interpolated_trajectory)

    def create_interpolated_trajectory(self, trajectory):
        """
        Create an interpolated trajectory from a given trajectory.
        :param trajectory: The trajectory to create the interpolated trajectory from.
        :return: The created interpolated trajectory.
        """
        waypoints = [Waypoint(point.time_point, point.oriented_box, point.velocity) for point in trajectory]
        return InterpolatedTrajectory(waypoints)

    def create_waypoint_from_point(self, point):
        """
        Create a waypoint from a point.
        :param point: The point to create the waypoint from.
        :return: The created waypoint.
        """
        pose = StateSE2(point.x, point.y, point.yaw)
        oriented_box = OrientedBox(
            pose,
            width=self.ego_vehicle_oriented_box.width,
            length=self.ego_vehicle_oriented_box.length,
            height=self.ego_vehicle_oriented_box.height
        )
        return Waypoint(
            time_point=point.time_point,
            oriented_box=oriented_box,
            velocity=point.velocity
        )
