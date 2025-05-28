from dataclasses import dataclass
from abc import ABC

@dataclass
class Observation(ABC):
    """
    Abstract observation container.
    """

    @classmethod
    def detection_type(cls) -> str:
        """
        Returns detection type of the observation.
        """
        return cls.__name__
