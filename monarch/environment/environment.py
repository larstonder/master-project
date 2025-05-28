from .abstract_environment import AbstractEnvironment
from ..types.state_types import SystemState, EnvState

class Environment(AbstractEnvironment):
    def __init__(self):
        pass

    def get_sensor_output(self, original_state: SystemState, last_state: SystemState, current_state: SystemState) -> EnvState:
        """
        Get the sensor output from the environment
        :param original_state: Original state of the environment
        :param last_state: Last state of the environment
        :param current_state: Current state of the environment
        :return: Sensor output
        """
        pass

if __name__=="__main__":
    print("INIT SANDBOX: ")