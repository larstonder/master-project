from abc import ABCMeta, abstractmethod

class AbstractSimulator(metaclass=ABCMeta):
    """Interface for class Simulator"""

    @abstractmethod
    def get_state(self):
        """Get the current state of the simulation."""
        pass

    @abstractmethod
    def do_action(self, trajectory):
        """Perform an action in the simulation."""
        pass
