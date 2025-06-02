"""
This script serves as the entry point for running a simulation of a vehicle environment.
It initializes the simulator, environment model,
and agent, and runs the simulation for a specified number of steps.
"""
import os
import math
from tqdm import tqdm
from rendering.abstract_renderer import AbstractRenderer
from rendering.omnire import OmniRe
from simulator.abstract_simulator import AbstractSimulator
from simulator.nuplan import NuPlan
from planning.abstract_planner import AbstractPlanner
from planning.oscillating_planner import OscillatingPlanner
from evaluation.abstract_evaluator import AbstractEvaluator
from evaluation.simple_evaluator import SimpleEvaluator
from scenario.abstract_scenario import AbstractScenario

from path_utils import use_path
from utils.image_utils import save_rgb_images_to_video

def do_simulation(
    n_steps: int,
    simulator: AbstractSimulator,
    renderer: AbstractRenderer,
    planner: AbstractPlanner,
    use_planner_input: bool = True
):
    """
    Run the simulation for a given number of steps.
    :param n_steps: Number of simulation steps to run.
    :param simulator: The simulator instance.
    :param renderer: The renderer instance.
    :param agent: The agent instance.
    :return: A list of error history.
    """

    sensor_outputs = []
    original_state = simulator.get_state()
    
    current_state = original_state
    
    for i in tqdm(range(n_steps)):
        last_state = current_state
        current_state = simulator.get_state()
                
        sensor_output = renderer.get_sensor_input(original_state, last_state, current_state, True)
        sensor_outputs.append(sensor_output)

        if use_planner_input:
            current_input = simulator.get_planner_input()
            trajectory = planner.compute_planner_trajectory(current_input)
        else:
            current_input = sensor_output
            trajectory = planner.compute_planner_trajectory(current_input)

        simulator.do_action(trajectory)
    
    return sensor_outputs

def main():
    """Main function to run the simulation."""
    
    renderer = None
    
    with use_path("drivestudio", True):
        relative_config_path = "configs/datasets/nuplan/8cams_undistorted.yaml"
        relative_checkpoint_path = "output/master-project/run_final"

        if not os.path.exists(relative_config_path):
            print(f"ERROR: Config file not found at {os.path.abspath(relative_config_path)}")
        if not os.path.exists(relative_checkpoint_path):
            print(f"ERROR: Checkpoint directory not found at {os.path.abspath(relative_checkpoint_path)}")

        if os.path.exists(relative_config_path) and os.path.exists(relative_checkpoint_path):
            renderer = OmniRe(relative_config_path, relative_checkpoint_path)
            print("Successfully initialized OmniRe environment model")
        else:
            print("Failed to initialize environment model due to missing files")

    simulator = NuPlan("2021.05.12.22.00.38_veh-35_01008_01518")
    planner = OscillatingPlanner(horizon_seconds=1.0, sampling_time=1.0, max_steering_angle=math.pi/8, steering_angle_increment=0.5)
    n_steps = 299

    results = do_simulation(n_steps, simulator, renderer, planner)
    save_rgb_images_to_video(results, "output_video.mp4")

if __name__ == "__main__":
    main()
