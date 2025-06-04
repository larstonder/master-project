from dataclasses import dataclass
from typing import Type, Optional

from abc import ABCMeta, abstractmethod
from monarch.typings.state_types import SystemState
from monarch.typings.trajectory import Trajectory

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
    def compute_planner_trajectory(self, current_input) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: List of planner inputs for which trajectory needs to be computed.
        :return: Trajectories representing the predicted ego's position in future
        """
        pass