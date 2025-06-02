from typing import Optional

from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import DrivableAreaComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance import DrivingDirectionComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_mean_speed import EgoMeanSpeedStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance import SpeedLimitComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import TimeToCollisionStatistics
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

from evaluator import Evaluator

class SimpleEvaluator(Evaluator):
    """
    Implements a predefined bare minimum evaluator
    Uses the base parameters specified in the test functions of each metric
    """
    def __init__(self, metric_score_unit: Optional[str] = None):
        lane_change_metric = EgoLaneChangeStatistics("lane_change", "Planning", 0.3)
        no_ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics(
            name="ego_at_fault_collision",
            category="Dynamics",
            ego_lane_change_metric=self.lane_change_metric,
            max_violation_threshold_vru=,0
            max_violation_threshold_vehicle=0,
            max_violation_threshold_object=1,
            metric_score_unit=metric_score_unit,
        ),
        metrics = [
            DrivableAreaComplianceStatistics(
                name="drivable_area_compliance",
                category="Planning",
                lane_change_metric=self.lane_change_metric,
                max_violation_threshold=0.3,
                metric_score_unit=metric_score_unit
            ),
            DrivingDirectionComplianceStatistics(
                name="driving_direction_compliance",
                category="Planning",
                lane_change_metric=self.lane_change_metric,
                driving_direction_compliance_threshold=2.0,
                driving_direction_violation_threshold=6.0,
                time_horizon=1.0,
                metric_score_unit=metric_score_unit
            ),
            no_ego_at_fault_collisions_metric,
            EgoMeanSpeedStatistics(
                name="ego_mean_speed",
                category="Planning"
            ),
            SpeedLimitComplianceStatistics(
                name="speed_limit_compliance",
                category="Planning",
                lane_change_metric=self.lane_change_metric,
                max_violation_threshold=1,
                max_overspeed_value_threshold=2.23,
                metric_score_unit=metric_score_unit
            ),
            TimeToCollisionStatistics(
                name="time_to_collision",
                category="Dynamics",
                ego_lane_change_metric=self.lane_change_metric,
                no_ego_at_fault_collision_metric=no_ego_at_fault_collisions_metric,
                time_step_size=1.0,
                time_horizon=1.0,
                least_min_ttc=3.0,
                metric_score_unit=metric_score_unit
            )
        ]
        super()__init__(name, metrics)

if __name__=="__main__":
    evaluator = SimpleEvaluator()
    evaluator.compute_cumulative_score()
