from typing import Type
from .abstract_planner import AbstractPlanner
from ..types.observation_type import Observation

class Planner(AbstractPlanner):
    def __init__(self, name: str, observation: Observation):
        self._name = name
        self._observation = observation

    @property
    def name(self):
        """
        Inherited, see superclass
        """
        return self._name

    @property
    def observation_type(self) -> Type[Observation]:
        """
        Inherited, see superclass
        """
        return self._observation

    def initialize(self, initialization) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization.map_api
        self._initialize_route_plan(initialization.route_roadblock_ids)
        self._initialized = False

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
        start_time = time.perf_counter()
        # If it raises an exception, catch to record the time then re-raise it.
        try:
            trajectory = self.compute_planner_trajectory(current_input)
        except Exception as e:
            self._compute_trajectory_runtimes.append(time.perf_counter() - start_time)
            raise e

        self._compute_trajectory_runtimes.append(time.perf_counter() - start_time)
        return trajectory

    def compute_planner_trajectory(self, current_input):
        """
        Inherited, see superclass
        """
        raise NotImplementedError("Function not implemented")