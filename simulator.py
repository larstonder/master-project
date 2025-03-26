import os
from omegaconf import OmegaConf
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
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

    def update(self):
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
                # 'num_workers': 4,  # Number of workers
                # '_target_': "nuplan.planning.utils.multithreading.worker_parallel.SingleMachineParallelExecutor",  # Specify the worker class
                # '_target_': 'nuplan.planning.script.builders.worker_pool.Sequental',  # Specify the worker class
            }
        })

        worker = build_worker(worker_config)
        scenario = scenario_builder.get_scenarios(scenario_filter, worker)[0]

        # print("Number of scenarios: ", len(scenario))

        time_controller = StepSimulationTimeController(scenario)
        observation = TracksObservation(scenario)
        # observations = build_observations(observation, scenario=scenario)
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

        print("NuPlan initialized.")

        planner_input = self.simulation.get_planner_input()
        
        # print planner input as a dictionary
        print(planner_input)

def main():
    nuplan = NuPlan()

if __name__ == "__main__":
    main()