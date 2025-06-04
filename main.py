"""
This script serves as the entry point for running a simulation of a vehicle environment.
It initializes the simulator, environment model,
and agent, and runs the simulation for a specified number of steps.
"""
import os
import math
from tqdm import tqdm
from monarch.rendering.abstract_renderer import AbstractRenderer
from monarch.rendering.omnire import OmniRe
from monarch.simulator.abstract_simulator import AbstractSimulator
from monarch.simulator.nuplan import NuPlan
from monarch.planning.abstract_planner import AbstractPlanner
from monarch.planning.oscillating_planner import OscillatingPlanner

from monarch.utils.path_utils import use_path
from monarch.utils.image_utils import save_rgb_images_to_video

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

    sensor_inputs = []
    original_state = simulator.get_state()
    
    state_history = [original_state]
    
    current_state = original_state
    
    for i in tqdm(range(n_steps)):
        last_state = current_state
        current_state = simulator.get_state()
        state_history.append(current_state)
        sensor_input = renderer.get_sensor_input(original_state, last_state, current_state, True)
        sensor_inputs.append(sensor_input)

        
        current_input = sensor_input
        trajectory = planner.compute_planner_trajectory(current_input, state_history)

        simulator.do_action(trajectory)
    
    return sensor_inputs

def main():
    """Main function to run the simulation."""
    
    renderer = None
    
    with use_path("./drivestudio", True):
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
