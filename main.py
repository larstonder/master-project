"""
This script serves as the entry point for running a simulation of a vehicle environment.
It initializes the simulator, environment model,
and agent, and runs the simulation for a specified number of steps.
"""

import os
import math
import numpy as np
import copy
from tqdm import tqdm
from monarch.rendering.abstract_renderer import AbstractRenderer
from monarch.rendering.omnire import OmniRe
from monarch.simulator.abstract_simulator import AbstractSimulator
from monarch.simulator.nuplan import NuPlan
from monarch.planning.abstract_planner import AbstractPlanner
from monarch.planning.simple_planner import SimplePlanner
from monarch.planning.oscillating_planner import OscillatingPlanner
from monarch.utils.path_utils import use_path
from monarch.utils.image_utils import save_rgb_images_to_video


def do_simulation(
    n_steps: int,
    simulator: AbstractSimulator,
    renderer: AbstractRenderer,
    planner: AbstractPlanner,
):
    """
    Run the simulation for a given number of steps.
    :param n_steps: Number of simulation steps to run.
    :param simulator: The simulator instance.
    :param renderer: The renderer instance.
    :param agent: The agent instance.
    """

    sensor_inputs = []
    original_state = simulator.get_state()

    state_history = [original_state]

    current_state = original_state

    for i in tqdm(range(n_steps)):
        last_state = current_state
        current_state = simulator.get_state()
        state_history.append(current_state)
        sensor_input = renderer.get_sensor_input(
            original_state, last_state, current_state, True
        )
        sensor_inputs.append(sensor_input)

        current_input = sensor_input
        trajectory = planner.compute_planner_trajectory(current_input, state_history)

        simulator.do_action(trajectory)

    return sensor_inputs

def experiment_1(simulator: AbstractSimulator, renderer: AbstractRenderer):
    """
    Experiment 1: Simple Planner
    """
    print("Experiment 1: Simple Planner")
    n_steps = 299
    planner = SimplePlanner(
        horizon_seconds=1.0,
        sampling_time=1.0,
        acceleration=np.array([5.0, 0.0]),
        max_velocity=10.0,
        steering_angle=0.0,
    )
    results = do_simulation(n_steps, simulator, renderer, planner)
    save_rgb_images_to_video(results, "experiments/experiment_1/experiment_1.mp4")
    
    simulator.reset()
    with use_path("./drivestudio", True):
        renderer.reset()

def experiment_2(simulator: AbstractSimulator, renderer: AbstractRenderer):
    """
    Experiment 2: Oscillating Planner
    """
    print("Experiment 2: Oscillating Planner")
    n_steps = 299
    planner = OscillatingPlanner(
        horizon_seconds=1.0,
        sampling_time=1.0,
        max_velocity=20.0,
        max_steering_angle=math.pi/8,
        acceleration=np.array([0.0, 0.0]),
        steering_angle_increment=0.5
    )
    
    results = do_simulation(n_steps, simulator, renderer, planner)
    save_rgb_images_to_video(results, "experiments/experiment_2/experiment_2.mp4")
    
    simulator.reset()
    with use_path("./drivestudio", True):
        renderer.reset()


def experiment_3(simulator: AbstractSimulator, renderer: AbstractRenderer):
    """
    Experiment 3: Editing vehicles
    """
    print("Experiment 3: Editing vehicles")
    n_steps_per_edit = 10
    sensor_outputs = []

    # Start with the initial state
    original_state = simulator.get_state()
    last_state = copy.deepcopy(simulator.get_state())
    current_state = copy.deepcopy(simulator.get_state())
    
    edits = ["x", "y", "z", "heading"]
    
    for edit in edits:
        sensor_outputs.append(renderer.get_sensor_input(original_state, last_state, current_state))
        
        for j in range(n_steps_per_edit):
            if edit == "x":
                current_state.vehicle_pos_list[0].x = current_state.vehicle_pos_list[0].x + (1 / n_steps_per_edit)
            elif edit == "y":
                current_state.vehicle_pos_list[0].y = current_state.vehicle_pos_list[0].y + (1 / n_steps_per_edit)
            elif edit == "z":
                current_state.vehicle_pos_list[0].z = current_state.vehicle_pos_list[0].z + (1 / n_steps_per_edit)
            elif edit == "heading":
                current_state.vehicle_pos_list[0].heading = current_state.vehicle_pos_list[0].heading + (1 / n_steps_per_edit)
            last_state = copy.deepcopy(current_state)
        
        sensor_outputs.append(renderer.get_sensor_input(original_state, last_state, current_state))
        save_rgb_images_to_video(sensor_outputs, f"experiments/experiment_3/exp_{edit}.mp4")
        
        with use_path("./drivestudio", True):
            renderer.reset()
    
    simulator.reset()
    with use_path("./drivestudio", True):
        renderer.reset()


def experiment_4(simulator: AbstractSimulator, renderer: AbstractRenderer):
    """
    Experiment 4: Edit ego vehicle position
    """
    print("Experiment 4: Edit ego vehicle position")
    
    original_state = simulator.get_state()
    last_state = copy.deepcopy(simulator.get_state())
    current_state = copy.deepcopy(simulator.get_state())
        
    edit_length = 2
    num_steps = 10
    
    edits = ["pos", "neg"]
    
    for edit in edits:
        sensor_outputs = []
        sensor_outputs.append(renderer.get_sensor_input(original_state, last_state, current_state))
        
        for i in range(num_steps):
            if edit == "pos":
                current_state.ego_pos.x = current_state.ego_pos.x + i * (edit_length / num_steps)
            elif edit == "neg":
                current_state.ego_pos.x = current_state.ego_pos.x - i * (edit_length / num_steps)
            sensor_outputs.append(renderer.get_sensor_input(original_state, last_state, current_state))
            last_state = copy.deepcopy(current_state)
        save_rgb_images_to_video(sensor_outputs, f"experiments/experiment_4/exp_{edit}.mp4")
        
        with use_path("./drivestudio", True):
            renderer.reset()

    simulator.reset()
    with use_path("./drivestudio", True):
        renderer.reset()

def run_experiments(renderer: AbstractRenderer, simulator: AbstractSimulator):
    """
    Run all experiments
    """    
    experiment_1(simulator, renderer)
    experiment_2(simulator, renderer)
    experiment_3(simulator, renderer)
    experiment_4(simulator, renderer)
    
def main():
    """Main function to run the simulation."""

    renderer = None

    with use_path("./drivestudio", True):
        relative_config_path = "configs/datasets/nuplan/8cams_undistorted.yaml"
        relative_checkpoint_path = "output/master-project/run_final"

        if not os.path.exists(relative_config_path):
            print(
                f"ERROR: Config file not found at {os.path.abspath(relative_config_path)}"
            )
        if not os.path.exists(relative_checkpoint_path):
            print(
                f"ERROR: Checkpoint directory not found at {os.path.abspath(relative_checkpoint_path)}"
            )

        if os.path.exists(relative_config_path) and os.path.exists(
            relative_checkpoint_path
        ):
            renderer = OmniRe(relative_config_path, relative_checkpoint_path)
            print("Successfully initialized OmniRe environment model")
        else:
            print("Failed to initialize environment model due to missing files")

    simulator = NuPlan("2021.05.12.22.00.38_veh-35_01008_01518")
    
    run_experiments(renderer, simulator)
    # planner = SimplePlanner(
    #     horizon_seconds=1.0,
    #     sampling_time=1.0,
    #     acceleration=np.array([5.0, 0.0]),
    #     max_velocity=10.0,
    #     steering_angle=0.0,
    # )
    # n_steps = 299

    # results = do_simulation(n_steps, simulator, renderer, planner)
    # save_rgb_images_to_video(results, "output_video.mp4")


if __name__ == "__main__":
    main()
