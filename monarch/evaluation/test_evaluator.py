from evaluator import Evaluator
from ..types.new_format.metric import Metric

class TestEvaluator(Evaluator):
    def __init__(self):
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

        raw_metrics = [
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

        metrics: List[Metric] = []
        for metric in raw_metrics:
            metrics.append(Metric(metric.name, metric.compute))
        super().__init__(self.__class__.__name__, metrics)
