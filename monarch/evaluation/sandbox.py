from nuplan.planning.metrics.abstract_metric import AbstractMetric
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evalutation_metrics.common.driveable_area_compliance import (
    DriveableAreaComplianceStatistics,
)

default_statistics_type_list = [
    MetricStatisticsType.MAX,
    MetricStatisticsType.MIN,
    MetricStatisticsType.P90,
    MetricStatisticsType.MEAN,
    MetricStatisticsType.VALUE,
    MetricStatisticsType.VELOCITY,
    MetricStatisticsType.BOOLEAN,
    MetricStatisticsType.RATIO,
    MetricStatisticsType.COUNT,
]

simple_eval_metrics = [
    "driveable_area_compliance",  # Checks if the car is within the legal boundaries of the environment (roads, lanes, intersections, etc.)
    "driving_direction_compliance",  # Checks if the car has been driving against the traffic flow more than some threshold.
    "ego_mean_speed",  # Returns average velocity
    "no_ego_at_fault_collisions",  # Since we use ego only systems instead of multi-agent systems
    "speed_limit_compliance",  # Checks if the vehicle keeps the speed limit
    "time_to_collision_within_bound",  # Calculates time to collision to other vehicles
]

rl_metrics = [
    "driveable_area_compliance",  # Checks if the car is within the legal boundaries of the environment (roads, lanes, intersections, etc.)
    "driving_direction_compliance",  # Checks if the car has been driving against the traffic flow more than some threshold.
    "ego_acceleration",  # Checks if the ego vehicle accelerates too fast
    "ego_stop_at_stop_line",  # Checks if the vehicle stops before the stop line
    "ego_jerk",  # change in acceleration
    "ego_lane_change",  # Statistics on lange change
    "ego_lat_acceleration",  # Metric for lateral acceleration (turning acceleration)
    "ego_lat_jerk",  # Lateral change in acceleration
    "ego_lon_acceleration",  # Normal acceleration and deceleration
    "ego_lon_jerk",  # Change in acceleration
    "ego_mean_speed",  # Returns average velocity
    "ego_yaw_acceleration",  # Computes how fast the turning of the wheel changes
    "ego_yaw_rate",  # turning rate of the vehicle
    "no_ego_at_fault_collisions",  # Since we use ego only systems instead of multi-agent systems
    "speed_limit_compliance",  # Checks if the vehicle keeps the speed limit
    "time_to_collision_within_bound",  # Calculates time to collision to other vehicles
]

il_metrics = [
    "ego_expert_l2_error_with_yaw",  # Euclidian distance between ego vehicle and expert vehicle with yaw angle (imitation learning)
    "ego_expert_l2_error",  # Euclidian distance between ego vehicle and expert vehicle (imitation learning)
    "ego_is_making_progress",  # Checks if ego trajectory is making progress along expert route
    "ego_progress_along_expert_route",  # computes progress along expert routes
    "planner_expert_average_heading_error_within_bound",  # Difference in absolute heading between ego agent and expert agent
    "planner_expert_average_l2_error_within_bound",  # Average l2 distance between ego vehicle and expert vehicle
    "planner_expert_final_heading_error_within_bound",  # Absoulte difference in heading at the final pose given a comparison time horizon
    "planner_expert_final_l2_error_within_bound",  # L2 error of planned pose w.r.t expert at the final pose given a comparison time horizon
    "planner_miss_rate_within_bound",  # Miss rate defined based on the maximum L2 error of planned ego pose w.r.t expert.
    "planner_miss_rate_within_bound",  # Miss rate defined based on the maximum L2 error of planned ego pose w.r.t expert.
]

all_metrics = [
    "driveable_area_compliance",  # Checks if the car is within the legal boundaries of the environment (roads, lanes, intersections, etc.)
    "driving_direction_compliance",  # Checks if the car has been driving against the traffic flow more than some threshold.
    "ego_acceleration",  # Checks if the ego vehicle accelerates too fast
    "ego_expert_l2_error_with_yaw",  # Euclidian distance between ego vehicle and expert vehicle with yaw angle (imitation learning)
    "ego_expert_l2_error",  # Euclidian distance between ego vehicle and expert vehicle (imitation learning)
    "ego_is_comfortable",  # Checks if passengers in ego vehicle is comfortable based on metrics such as acceleration, jerk, etc.
    "ego_is_making_progress",  # Checks if ego trajectory is making progress along expert route
    "ego_jerk",  # change in acceleration
    "ego_lane_change",  # Statistics on lange change
    "ego_lat_acceleration",  # Metric for lateral acceleration (turning acceleration)
    "ego_lat_jerk",  # Lateral change in acceleration
    "ego_lon_acceleration",  # Normal acceleration and deceleration
    "ego_lon_jerk",  # Change in acceleration
    "ego_mean_speed",  # Returns average velocity
    "ego_progress_along_expert_route",  # computes progress along expert routes
    "ego_yaw_acceleration",  # Computes how fast the turning of the wheel changes
    "ego_yaw_rate",  # turning rate of the vehicle
    "no_ego_at_fault_collisions",  # Since we use ego only systems instead of multi-agent systems
    "planner_expert_average_heading_error_within_bound",  # Difference in absolute heading between ego agent and expert agent
    "planner_expert_average_l2_error_within_bound",  # Average l2 distance between ego vehicle and expert vehicle
    "planner_expert_final_heading_error_within_bound",  # Absoulte difference in heading at the final pose given a comparison time horizon
    "planner_expert_final_l2_error_within_bound",  # L2 error of planned pose w.r.t expert at the final pose given a comparison time horizon
    "planner_miss_rate_within_bound",  # Miss rate defined based on the maximum L2 error of planned ego pose w.r.t expert.
    "speed_limit_compliance",  # Checks if the vehicle keeps the speed limit
    "time_to_collision_within_bound",  # Calculates time to collision to other vehicles
    "ego_stop_at_stop_line",  # Checks if the vehicle stops before the stop line
]


class SimpleEval:

    def __init__(
        self,
        scenario: AbstractScenario,
        statistics_type_list: Optional[
            list[MetricStatisticsType]
        ] = default_statistics_type_list,
    ):
        self.name = "SimpleEval"
        self.scenario = scenario
        self.statistics_type_list = statistics_type_list


if __name__ == "__main__":
    test_eval = SimpleEval()
    print(test_eval.name)
