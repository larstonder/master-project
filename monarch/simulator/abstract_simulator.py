from abc import ABCMeta, abstractmethod
from ..types.trajectory import Trajectory
from ..types.scenario import Scenario
from ..types.simulation import Simulation
from ..types.state_types import SystemState

class AbstractSimulator(metaclass=ABCMeta):
    """Interface for class Simulator"""
    @property
    @abstractmethod
    def scenario(self) -> Scenario:
        """Get the scenario of the simulation."""
        pass

    @property
    @abstractmethod
    def simulation(self) -> Simulation:
        """Get the simulation instance."""
        pass

    @abstractmethod
    def get_state(self) -> SystemState:
        """Get the current state of the simulation."""
        pass

    @abstractmethod
    def do_action(self, trajectory: Trajectory):
        """Perform an action in the simulation."""
        pass
