from dataclasses import dataclass
from state_types import State

@dataclass
class PlannerInitializationParams:
    """
    Dataclass for initializing more advanced planners. Can be expanded upon as needed.
    """
    mission_goal: State
