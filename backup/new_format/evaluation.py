from dataclasses import dataclass
# Move to evaluation package?
class Metric:
    def __init__(self, name: str, compute):
        self._name = name
        self._compute = compute

    @property
    def name(self) -> str:
        """
        Return the name of the metric.
        :return: Metric name
        """
        return self._name
    
    def compute(self, history, scenario) -> list[float]:
        return self._compute(history, scenario)
