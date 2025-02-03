from eval import get_evaluation
from simulator import Simulator
from environment_model import EnvironmentModel
from agent import Agent
import jax

@jax.jit
def do_simulation(
    n_steps: int,
    simulator: Simulator,
    environment_model: EnvironmentModel,
    agent: Agent
):
    error_history = []

    for _ in range(n_steps):
        state = simulator.get_state()
        sensor_output = environment_model.get_sensor_output(state)
        action = agent.get_action(sensor_output)
        simulator.do_action(action)
        error_history.append(simulator.get_state()) # something like this
        simulator.update()
    
    return error_history

def main():
    simulator = Simulator()
    environment_model = EnvironmentModel()
    agent = Agent()
    n_steps = 100

    do_simulation(n_steps, simulator, environment_model, agent)

if __name__ == "__main__":
    main()