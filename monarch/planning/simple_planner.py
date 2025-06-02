from .planner import Planner
from monarch.types.trajectory import Trajectory
from monarch.types.planner_input import PlannerInput
import numpy as np

class SimplePlanner(Planner):
    """
    Planner going straight.
    """

    def __init__(self, horizon_seconds: float, sampling_time: float, acceleration: np.array, max_velocity: float, steering_angle: float):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param acceleration: [m/s^2] acceleration of the vehicle.
        :param max_velocity: [m/s] max velocity of the vehicle.
        :param steering_angle: [rad] steering angle of the vehicle.
        """
        super().__init__(self.__class__.__name__)
        self.horizon_seconds = horizon_seconds
        self.sampling_time = sampling_time
        self.acceleration = acceleration
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        
    def compute_planner_trajectory(self, current_input: PlannerInput) -> Trajectory:
        """
        Computes the trajectory of the vehicle.
        :param current_input: [PlannerInput] current input of the vehicle.
        :return: [Trajectory] trajectory of the vehicle.
        """
        """
        Computes the trajectory of the vehicle driving straight ahead.
        :param current_input: [PlannerInput] current input of the vehicle.
        :return: [Trajectory] trajectory of the vehicle.
        """
        # Get the current state
        history = current_input.history
        ego_state = history.ego_states[-1]
        x, y = ego_state.rear_axle.x, ego_state.rear_axle.y
        heading = ego_state.rear_axle.heading
        velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.x

        # Prepare trajectory
        trajectory = []
        time_steps = int(self.horizon_seconds / self.sampling_time)
        dt = self.sampling_time

        for i in range(time_steps):
            # Accelerate up to max_velocity
            velocity = min(velocity + self.acceleration[0] * dt, self.max_velocity)
            # Move straight ahead
            x += velocity * np.cos(heading) * dt
            y += velocity * np.sin(heading) * dt
            # Always use the same heading (straight)
            trajectory.append(Waypoint(x, y, heading))

        return Trajectory(trajectory)
        