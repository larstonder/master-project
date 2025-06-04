from monarch.simulator.abstract_simulator import AbstractSimulator
from monarch.typings.state_types import SystemState

class Simulator(AbstractSimulator):
    def __init__(self, scenario, simulation): # Define scenario and simulation as classes in types folder
        self._scenario = scenario
        self._simulation = simulation

    @property
    def scenario(self):
        return self._scenario

    @property
    def simulation(self):
        return self._simulation
    
    def reset(self):
        """Inherited, see superclass"""
        raise NotImplementedError("This method should be overridden in subclasses.")

    def get_state(self) -> SystemState: # Define SystemState as a class in types folder
        """Inherited, see superclass"""
        raise NotImplementedError("This method should be overridden in subclasses.")

    def do_action(self, trajectory): # Define trajectory as a class in types folder
        """Inherited, see superclass"""
        raise NotImplementedError("This method should be overridden in subclasses.")
