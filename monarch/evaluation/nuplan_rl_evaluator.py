# Add to the top of train.py after existing imports:
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

class NuPlanRLEvaluator:
    """
    Hybrid RL evaluator that combines strong forward progress rewards 
    with sophisticated NuPlan metrics for realistic driving evaluation.
    """
    
    def __init__(self):
        # Initialize NuPlan metrics (with proper error handling)
        self.nuplan_metrics = self._initialize_nuplan_metrics()
        
        # Forward progress tracking
        self.previous_position = None
        self.episode_start_position = None
        self.total_distance_traveled = 0.0
        self.trajectory_count = 0
        
        # Mock scenario for NuPlan metrics
        self.mock_scenario = MockAbstractScenario()
        
    def _initialize_nuplan_metrics(self):
        """Initialize core NuPlan metrics that are most relevant for RL training."""
        try:
            from nuplan.planning.metrics.evaluation_metrics.common.ego_mean_speed import EgoMeanSpeedStatistics
            from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import DrivableAreaComplianceStatistics
            from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance import DrivingDirectionComplianceStatistics
            from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
            
            # Initialize key metrics
            lane_change_metric = EgoLaneChangeStatistics("lane_change", "Planning", 0.3)
            
            metrics = {
                'speed': EgoMeanSpeedStatistics("ego_mean_speed", "Planning"),
                'lane_change': lane_change_metric,
                'drivable_area': DrivableAreaComplianceStatistics(
                    "drivable_area_compliance", "Planning", 
                    lane_change_metric=lane_change_metric, 
                    max_violation_threshold=0.3
                ),
                'driving_direction': DrivingDirectionComplianceStatistics(
                    "driving_direction_compliance", "Planning",
                    lane_change_metric=lane_change_metric,
                    driving_direction_compliance_threshold=2.0,
                    driving_direction_violation_threshold=6.0,
                    time_horizon=1.0
                )
            }
            
            print("Successfully initialized NuPlan metrics for RL evaluation")
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not initialize NuPlan metrics: {e}")
            return {}
    
    def compute_cumulative_score(self, trajectory_history, current_state):
        """
        Compute hybrid reward combining strong forward progress incentives with NuPlan metrics.
        """
        if len(trajectory_history) == 0:
            return 0.0
        
        total_reward = 0.0
        
        # **1. STRONG FORWARD PROGRESS REWARDS** (Primary driver - 70% weight)
        progress_reward = self._compute_forward_progress_reward(current_state)
        total_reward += progress_reward * 0.7
        
        # **2. NUPLAN METRIC REWARDS** (Quality assessment - 30% weight)  
        if self.nuplan_metrics:
            try:
                nuplan_reward = self._compute_nuplan_reward(trajectory_history, current_state)
                total_reward += nuplan_reward * 0.3
            except Exception as e:
                print(f"NuPlan evaluation error: {e}")
                # Add basic quality bonus if NuPlan fails
                total_reward += self._compute_basic_quality_reward(trajectory_history[-1]) * 0.3
        else:
            # Fallback to basic quality rewards
            total_reward += self._compute_basic_quality_reward(trajectory_history[-1]) * 0.3
        
        return total_reward
    
    def _compute_forward_progress_reward(self, current_state):
        """Compute strong rewards for forward progress and movement."""
        reward = 0.0
        
        # Extract current position
        current_pos = current_state.ego_pos
        current_x = getattr(current_pos, 'x', 0.0)
        current_y = getattr(current_pos, 'y', 0.0)
        
        # Initialize episode tracking
        if self.episode_start_position is None:
            self.episode_start_position = (current_x, current_y)
        
        # **DISTANCE MOVED REWARD** (primary driver)
        if self.previous_position is not None:
            prev_x, prev_y = self.previous_position
            distance_moved = math.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)
            
            # Major reward for distance traveled
            distance_reward = distance_moved * 20.0  # 20 points per meter moved
            reward += distance_reward
            
            # Bonus for consistent movement
            if distance_moved > 0.5:  # Good movement
                reward += 25.0
            elif distance_moved > 0.1:  # Some movement
                reward += 10.0
            else:  # Very little movement
                reward -= 15.0  # Penalty for staying still
        
        # **CUMULATIVE DISTANCE FROM START REWARD**
        if self.episode_start_position is not None:
            start_x, start_y = self.episode_start_position
            total_displacement = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            
            # Progressive reward for getting farther from start
            displacement_reward = total_displacement * 3.0  # 3 points per meter of displacement
            reward += displacement_reward
        
        # **SPEED REWARD**
        if hasattr(current_state.ego_pos, 'vehicle_parameters'):
            vx = current_state.ego_pos.vehicle_parameters.vx
            vy = current_state.ego_pos.vehicle_parameters.vy
            speed = math.sqrt(vx**2 + vy**2)
            
            # Strong reward for good driving speeds
            if 3.0 <= speed <= 12.0:  # Optimal driving speed
                speed_reward = 30.0 + (speed * 3.0)  # Up to 66 points for good speed
                reward += speed_reward
            elif 1.0 <= speed < 3.0:  # Slow but moving
                reward += 15.0
            elif speed < 0.5:  # Nearly stopped
                reward -= 25.0  # Strong penalty
            elif speed > 15.0:  # Too fast
                reward -= 15.0
            
            # Forward motion bonus
            if vx > 0:
                reward += 10.0  # Bonus for forward direction
        
        # Update tracking
        self.previous_position = (current_x, current_y)
        self.trajectory_count += 1
        
        return reward
    
    def _compute_nuplan_reward(self, trajectory_history, current_state):
        """Compute rewards using NuPlan metrics."""
        try:
            # Convert our data to NuPlan format
            simulation_history = self._create_simulation_history(trajectory_history, current_state)
            
            total_nuplan_reward = 0.0
            
            # Evaluate each metric and convert to reward
            for metric_name, metric in self.nuplan_metrics.items():
                try:
                    # Most NuPlan metrics return violations (lower is better)
                    # We need to convert these to rewards (higher is better)
                    metric_result = metric.compute(simulation_history, self.mock_scenario)
                    
                    if metric_name == 'speed':
                        # Speed metric: reward for reasonable speeds
                        if 2.0 <= metric_result <= 10.0:
                            total_nuplan_reward += 20.0
                        elif metric_result > 0.5:
                            total_nuplan_reward += 10.0
                    
                    elif metric_name in ['drivable_area', 'driving_direction']:
                        # Compliance metrics: reward for fewer violations
                        violation_penalty = metric_result * 10.0  # Scale violations
                        total_nuplan_reward -= violation_penalty
                        
                        # Bonus for no violations
                        if metric_result == 0:
                            total_nuplan_reward += 15.0
                    
                    elif metric_name == 'lane_change':
                        # Lane change: small penalty for frequent changes
                        total_nuplan_reward -= metric_result * 5.0
                    
                except Exception as e:
                    print(f"Error computing {metric_name}: {e}")
                    continue
            
            return total_nuplan_reward
            
        except Exception as e:
            print(f"Error in NuPlan reward computation: {e}")
            return 0.0
    
    def _create_simulation_history(self, trajectory_history, current_state):
        """Convert our trajectory data to NuPlan SimulationHistory format."""
        # This is a simplified conversion - in practice you'd want more sophisticated mapping
        history = SimulationHistory(self.mock_scenario.map_api, self.mock_scenario.get_mission_goal())
        
        for i, trajectory in enumerate(trajectory_history[-5:]):  # Use last 5 trajectories
            if len(trajectory.waypoints) > 0:
                # Convert to EgoState
                waypoint = trajectory.waypoints[0]
                ego_state = EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(waypoint.x, waypoint.y, waypoint.heading),
                    rear_axle_velocity_2d=StateVector2D(waypoint.vx, waypoint.vy),
                    rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                    tire_steering_angle=0.0,
                    time_point=TimePoint(int(waypoint.timestamp)),
                    vehicle_parameters=self.mock_scenario.ego_vehicle_parameters
                )
                
                # Create trajectory for this iteration
                interpolated_trajectory = InterpolatedTrajectory([ego_state])
                
                # Add to history
                iteration = SimulationIteration(TimePoint(int(waypoint.timestamp)), i)
                sample = SimulationHistorySample(
                    iteration=iteration,
                    ego_state=ego_state,
                    trajectory=interpolated_trajectory,
                    observation=DetectionsTracks(TrackedObjects()),
                    traffic_light_status=[]
                )
                history.add_sample(sample)
        
        return history
    
    def _compute_basic_quality_reward(self, trajectory):
        """Fallback quality rewards when NuPlan metrics aren't available."""
        reward = 0.0
        
        if len(trajectory.waypoints) >= 2:
            # Smoothness reward
            total_heading_change = 0.0
            for i in range(1, len(trajectory.waypoints)):
                heading_diff = abs(trajectory.waypoints[i].heading - trajectory.waypoints[i-1].heading)
                if heading_diff > math.pi:
                    heading_diff = 2 * math.pi - heading_diff
                total_heading_change += heading_diff
            
            avg_heading_change = total_heading_change / (len(trajectory.waypoints) - 1)
            if avg_heading_change < 0.1:
                reward += 15.0  # Very smooth
            elif avg_heading_change < 0.3:
                reward += 8.0   # Moderately smooth
            else:
                reward -= 5.0   # Too jerky
        
        return reward
    
    def reset_episode(self):
        """Reset for new episode."""
        self.previous_position = None
        self.episode_start_position = None
        self.total_distance_traveled = 0.0
        self.trajectory_count = 0