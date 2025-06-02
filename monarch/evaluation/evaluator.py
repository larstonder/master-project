from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from abstract_evaluator import AbstractEvaluator
from ..types.new_format.metric import Metric


class Evaluator(AbstractEvaluator):
    """
    Base class for all evaluators
    """

    def __init__(self, name, metrics):
        self._name = name
        self._metrics = metrics

    @property
    def name(self) -> str:
        """
        Return the name of the evaluator.
        :return evaluator name
        """
        return self._name

    @property
    def metrics(self) -> List[Metric]:
        """
        Return the different metric statistics implemented in the evaluator
        :return list of different metricStatistics
        """
        return self._metrics

    def compute_cumulative_score(self, history, scenario) -> float:
        """
        
        """
        cumulative_score: float = 0.0
        for metric in self._metrics:
            cumulative_score += metric.compute(history, scenario)

        return cumulative_score
