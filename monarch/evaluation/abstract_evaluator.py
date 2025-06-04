from abc import ABCMeta, abstractmethod
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from typing import List
from monarch.typings.metric import Metric


class AbstractEvaluator(metaclass=ABCMeta):
    """Interface for generic evaluator"""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the evaluator.
        :return evaluator name
        """
        pass

    @property
    @abstractmethod
    def metrics(self) -> List[Metric]:
        """
        Return the different metric statistics implemented in the evaluator
        :return list of different metricStatistics
        """
        pass

    @abstractmethod
    def compute_cumulative_score(self, history, scenario) -> float:
        """
        Calculate the total evaluation of the current state of the system
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return The score of the current state
        """
        pass
