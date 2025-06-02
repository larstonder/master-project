from typing import Type
from .abstract_planner import AbstractPlanner
from ..types.trajectory import Trajectory

class Planner(AbstractPlanner):
    def __init__(self, name: str, initialization_params = None): # Might need to remove Observation
        """
        Constructor for Planner
        :param name: Name of the planner
        :param initialization_params: Parameters for more advanced planners
        """
        self._name = name
        if initialization_params:
            self._mission_goal = initialization_params.mission_goal

    @property
    def name(self):
        """
        Inherited, see superclass
        """
        return self._name


    def compute_planner_trajectory(self, current_input) -> Trajectory:
        """
        Inherited, see superclass
        """
        raise NotImplementedError("Function not implemented")
