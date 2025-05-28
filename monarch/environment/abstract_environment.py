from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

class AbstractEnvironment(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        
        """
        pass