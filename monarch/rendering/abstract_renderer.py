from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from monarch.typings.state_types import SystemState, EnvState

class AbstractRenderer(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the renderer
        :return: Name of the renderer
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the renderer
        """
        pass
    
    @abstractmethod
    def get_sensor_input(self, original_state: SystemState, last_state: SystemState, current_state: SystemState) -> EnvState:
        """
        Get the sensor output from the environment
        :param original_state: Original state of the environment
        :param last_state: Last state of the environment
        :param current_state: Current state of the environment
        :return: Sensor output
        """
        pass