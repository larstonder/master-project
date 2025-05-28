"""
This file contains the definition of the Agent class, which is responsible for
interacting with the simulation environment. The Agent class is designed to be
used in conjunction with the Simulator and EnvironmentModel classes to
perform actions based on sensor data from the simulation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    TimePoint,
    StateVector2D,
)
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.waypoint import Waypoint
from interfaces import Agent
from nn import MLP
from state_types import Action, EnvState, SystemState


class SSRLAgent(Agent):
    """SuperSimple Reinforcement Learning agent meant for an easy benchmark againt the RandomAgent."""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters = {
            "input_channels": 3,
            "output_dim": 2,
            "hidden_dims": [16, 32],
            "img_size": (360, 640), # Input from sensor_output
            "lin_width": 230400,
            "batch_size": 1,
            "num_epochs": 1,
            "lr": 1e-3
        }
        self.nn = MLP(
            self.hyperparameters["input_channels"],
            self.hyperparameters["output_dim"],
            self.hyperparameters["hidden_dims"],
            self.hyperparameters["lin_width"]
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.hyperparameters["lr"])

    def act(self, sensor_outputs: list[EnvState]) -> list[Action]:
        sensor_output = sensor_outputs[0] # 
        # Ensure correct format on sensor_output
        action = self.nn(sensor_output.rgb)
        action = Action(action)
        return action

    def compute_loss(self, state: SystemState, action: Action) -> torch.Tensor:
        # Placeholder
        pass

class RLAgent(Agent):
    """
    Reinforcement Learning agent utilising all aspects of the environment and simulator.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters = {
            "input_channels": 3,
            "output_dim": 2,
            "hidden_dims": [16, 32],
            "img_size": (32, 32),
            "batch_size": 1,
            "num_epochs": 1,
            "lr": 1e-3
        }
        self.nn = MLP(
            hyperparameters["input_channels"],
            hyperparameters["output_dim"],
            hyperparameters["hidden_dims"],
            hyperparameters["img_size"][0] * hyperparameters["img_size"][1]
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.hyperparameters["lr"])
    
    def act(self, sensor_outputs: list[EnvState]) -> Action:
        """
        Returns an action for a given sensor output from the environment.
        The function is called for each iteration of the environment.
        :param sensor_output: RGB image from the environment
        :return: Action to be performed by the agent
        """

class RandomAgent(Agent):
    def act(self, sensor_outputs: list[EnvState]) -> Action:
        acceleration = random.uniform(-1, 1)
        steering_angle = random.uniform(-1, 1)
        return Action(acceleration, steering_angle)

    def compute_loss(self):
        pass

class BackupRandomAgent(Agent):
    """Random agent that selects random actions."""

    def get_action(self, sensor_output, timestamp: TimePoint):
        """
        Select a random trajectory
        :param sensor_output: RGB image from the environment
        :param timestamp: Timestamp of the current simulation step
        """
        trajectory: list[Waypoint] = []
        current_time = timestamp.time_us

        for i in range(10):
            # Create proper TimePoint object (microseconds)
            time_point = TimePoint(current_time + i * 100000)

            # Create position and heading
            x = i * 10
            y = i * 10
            heading = -2.066

            # Create StateSE2 for position and heading
            center = StateSE2(x, y, heading)

            # Create oriented box
            # Parameters: center, length, width, height
            oriented_box = OrientedBox(center, length=1.0, width=1.0, height=1.0)

            # Create velocity vector
            velocity = StateVector2D(1.0, 0.0)  # x-velocity=1.0, y-velocity=0.0

            # Create waypoint with all required components
            waypoint = Waypoint(time_point, oriented_box, velocity)
            trajectory.append(waypoint)
            with open("output.txt", "w") as f:
                f.write(f"Time point: {time_point}\n")
                f.write(f"Center: {center}\n")
                f.write(f"Oriented box: {oriented_box}\n")
                f.write(f"Velocity: {velocity}\n")
                f.write(f"Waypoint: {waypoint}\n")

        return trajectory
