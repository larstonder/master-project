import os
from omegaconf import OmegaConf
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
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
    def __init__(self):
        pass
    
    def get_state(self):
        pass

    def do_action(self, action):
        pass

class NuPlan(Simulator):

    def __init__(self):
        print("Initializing NuPlan...")
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
                '_target_': 'nuplan.planning.utils.multithreading.worker_sequential.Sequential',  # Specify the worker class
            }
        })

        worker = build_worker(worker_config)
        scenario = scenario_builder.get_scenarios(scenario_filter, worker)[0]

        time_controller = StepSimulationTimeController(scenario)
        observation = TracksObservation(scenario)
        controller = PerfectTrackingController(scenario)

        simulation_setup = SimulationSetup(
            time_controller=time_controller,  # Required simulation time controller
            observations=observation,  # Required observation
            ego_controller=controller,  # Required ego controller
            scenario=scenario  # Required scenario
        )
        
        self.simulation = Simulation(
            simulation_setup=simulation_setup,  # Required simulation setup
            callback=None,  # Optional callback functions
            simulation_history_buffer_duration=2.0  # Optional history buffer duration
        )

        self.simulation.initialize()
        
        planner_input = self.simulation.get_planner_input()
        history = planner_input.history
        self.original_ego_state, self.original_observation_state = history.current_state
        
        self.ego_vehicle_oriented_box = self.original_ego_state[-1].waypoint.oriented_box

        print("NuPlan initialized.")
    
    def get_state(self):
        planner_input = self.simulation.get_planner_input()
        history = planner_input.history
        ego_state, observation_state = history.current_state
        
        ego_pos = ego_state[-1].waypoint.center
        agent_pos_list = [
            agent.center for agent in observation_state[-1].tracked_objects.get_agents()
        ]
        return ego_pos, agent_pos_list
    
    def do_action(self, action):
        trajectory = action.get_trajectory()
        interpolated_trajectory = self.create_interpolated_trajectory(trajectory)
        self.simulation.propagate(interpolated_trajectory)
    
    def create_interpolated_trajectory(self, trajectory):
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
            length=width=self.ego_vehicle_oriented_box.length,
            height=width=self.ego_vehicle_oriented_box.height
        )
        return Waypoint(
            time_point=point.time_point,
            oriented_box=point.oriented_box,
            velocity=point.velocity
        )

def main():
    nuplan = NuPlan()

if __name__ == "__main__":
    main()