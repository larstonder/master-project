from .abstract_renderer import AbstractRenderer
from ..types.state_types import SystemState, EnvState

class Renderer(AbstractRenderer):
    def __init__(self):
        pass

    def get_sensor_input(self, original_state: SystemState, last_state: SystemState, current_state: SystemState) -> EnvState:
        """Inherited, see superclass"""
        raise NotImplementedError("This method should be overridden in subclasses.")

if __name__=="__main__":
    print("INIT SANDBOX: ")