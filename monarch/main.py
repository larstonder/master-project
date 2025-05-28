"""
This script serves as the entry point for running a simulation of a vehicle environment.
It initializes the simulator, environment model,
and agent, and runs the simulation for a specified number of steps.
"""
import os
from environment.abstract_environment import AbstractEnvironment
from environment.omnire import OmniRe, OmniReSetup
from simulator.abstract_simulator import AbstractSimulator
from simulator.nuplan import NuPlan
from planning.abstract_planner import AbstractPlanner
from planning.rl_planner import ReinforcementLearningPlanner
from evaluation.abstract_evaluator import AbstractEvaluator
from evaluation.simple_evaluator import SimpleEvaluator
from scenario.abstract_scenario import AbstractScenario # TODO: Add scenario class

from path_utils import use_path
from utils.image_utils import save_rgb_images_to_video

def do_simulation(
    n_steps: int,
    simulator: AbstractSimulator,
    environment_model: AbstractEnvironment,
    agent: AbstractPlanner
):
    """
    Run the simulation for a given number of steps.
    :param n_steps: Number of simulation steps to run.
    :param simulator: The simulator instance.
    :param environment_model: The environment model instance.
    :param agent: The agent instance.
    :return: A list of error history.
    """

    for _ in range(n_steps):
        state = simulator.get_state()
        sensor_output = environment_model.get_sensor_output(state)
        action = agent.get_action(sensor_output)
        simulator.do_action(action)
        error_history.append(simulator.get_state())
    return error_history

def train_planner(
    n_steps: int,
    simulator: AbstractSimulator,
    environment_model: AbstractEnvironment,
    agent: AbstractPlanner,
    evaluator: AbstractEvaluator,
    scenario: AbstractScenario
):
    """
    Train the planner for a given number of steps.
    """
    for _ in range(n_steps):
        state = simulator.get_state()
        sensor_output = environment_model.get_sensor_output(state)
        action = agent.get_action(sensor_output)
        simulator.do_action(action)
        
        history = simulator.get_history()
        evaluator.compute_cumulative_score(history, scenario)
        
        
        
        
        
    return evaluator.get_results()
    

def main():
    """Main function to run the simulation."""
    env_setup = None

    # Now use the context manager for clean path handling
    with use_path("drivestudio", True):
        # Define paths relative to the drivestudio directory
        relative_config_path = "configs/datasets/nuplan/8cams_undistorted.yaml"
        relative_checkpoint_path = "output/master-project/run_omnire_undistorted_8cams_0"

        print(f"Working directory: {os.getcwd()}")
        print(f"Config path (relative to drivestudio): {relative_config_path}")
        print(f"Absolute config path: {os.path.abspath(relative_config_path)}")

        # Check if these files exist in this context
        if not os.path.exists(relative_config_path):
            print(f"ERROR: Config file not found at {os.path.abspath(relative_config_path)}")
        if not os.path.exists(relative_checkpoint_path):
            print(f"ERROR: Checkpoint directory not found at {os.path.abspath(relative_checkpoint_path)}")

        # Only initialize if files exist
        if os.path.exists(relative_config_path) and os.path.exists(relative_checkpoint_path):
            env_setup = OmniReSetup(relative_config_path, relative_checkpoint_path)
            print("Successfully initialized OmniRe environment model")
        else:
            print("Failed to initialize environment model due to missing files")

    simulator = NuPlan()
    environment_model = OmniRe(env_setup)
    agent = RandomAgent()
    n_steps = 100

    results = do_simulation(n_steps, simulator, environment_model, agent)
    save_rgb_images_to_video(results, "output_video.mp4")

if __name__ == "__main__":
    main()
