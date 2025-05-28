from dataclasses import dataclass
from typing import Type, Optional

from abc import ABCMeta, abstractmethod
from ..types.observation_type import Observation
from ..types.state_types import SystemState

class AbstractPlanner(metaclass=ABCMeta):
    """Interface for all planners"""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the planner.
        :return planner name
        """
        pass

    @abstractmethod
    def initialize(self, initialization) -> None:
        """
        Initialize planner
        :param initialization: instance of initialization
        """
        pass

    @abstractmethod
    def observation_type(self) -> Type[Observation]:
        """
        :return Type of observation that is expected in compute_trajectory
        """
        pass

    def compute_trajectory(self, current_input):
        """
        Computes the ego vehicle trajectory, where we check that if planner can not consume batched inputs,
            we require that the input list has exactly one element
        :param current_input: List of planner inputs for where for each of them trajectory should be computed
            In this case the list represents batched simulations. In case consume_batched_inputs is False
            the list has only single element
        :return: Trajectories representing the predicted ego's position in future for every input iteration
            In case consume_batched_inputs is False, return only a single trajectory in a list.
        """
        pass

    @abstractmethod
    def compute_planner_trajectory(self, current_input):
        """
        Computes the ego vehicle trajectory.
        :param current_input: List of planner inputs for which trajectory needs to be computed.
        :return: Trajectories representing the predicted ego's position in future
        """
        pass