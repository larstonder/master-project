"""
This script serves as the entry point for running a simulation of a vehicle environment.
It initializes the simulator, environment model,
and agent, and runs the simulation for a specified number of steps.
"""
from simulator import Simulator, NuPlan
from environment_model import EnvironmentModel, OmniRe
from agent import Agent, RandomAgent

def do_simulation(
    n_steps: int,
    simulator: Simulator,
    environment_model: EnvironmentModel,
    agent: Agent
):
    """
    Run the simulation for a given number of steps.
    :param n_steps: Number of simulation steps to run.
    :param simulator: The simulator instance.
    :param environment_model: The environment model instance.
    :param agent: The agent instance.
    :return: A list of error history.
    """
    error_history = []

    for _ in range(n_steps):
        state = simulator.get_state()
        sensor_output = environment_model.get_sensor_output(state)
        action = agent.get_action(sensor_output)
        simulator.do_action(action)
        error_history.append(simulator.get_state()) # something like this
        # if step % 10 == 0:
        #     # Move vehicle 1 every 10 steps
        #     new_position = np.array([2.0 + step/10, 0.0, 0.0])
        #     state = env_model.update_vehicle_positions(
        #         state, 
        #         vehicle_id=1,
        #         new_position=new_position,
        #         new_rotation=state["vehicle_rotations"][1]
        #     )
    return error_history

def main():
    """Main function to run the simulation."""
    config_path = "configs/datasets/nuplan/8cams_undistorted.yaml"
    checkpoint_path = "output/master-project/run_omnire_undistorted_8cams_0"

    simulator = NuPlan()
    environment_model = OmniRe(config_path, checkpoint_path)
    agent = RandomAgent()
    n_steps = 100

    do_simulation(n_steps, simulator, environment_model, agent)

if __name__ == "__main__":
    main()
