from abc import ABCMeta, abstractmethod
from monarch.typings.trajectory import Trajectory
from monarch.typings.scenario import Scenario
from monarch.typings.simulation import Simulation
from monarch.typings.state_types import SystemState

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
    def reset(self):
        """Reset the simulator."""
        pass

    @abstractmethod
    def get_state(self) -> SystemState:
        """Get the current state of the simulation."""
        pass

    @abstractmethod
    def do_action(self, trajectory: Trajectory):
        """Perform an action in the simulation."""
        pass
