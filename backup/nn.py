"""
Neural Network Module
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Basic MLP
    Accepts inputs of (3, )
    """

    def __init__(self, input_dim, output_dim, hidden_dims, lin_width):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for current_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels=prev_dim, out_channels=current_dim, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            prev_dim = current_dim

        layers.append(nn.Flatten())
        layers.append(nn.Linear(lin_width, output_dim))

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        print("Before:", x.shape)
        x = self.nn(x)
        print("After:", x.shape)

if __name__ == "__main__":
    nn = MLP(3, 2, [16, 32])
    x = torch.zeros(3, 360, 640)
    print(x.shape)
    x = nn.forward(x)
    print("In main", x.shape)
