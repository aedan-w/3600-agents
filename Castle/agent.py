from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple
import os

import numpy as np
from game import *
import game.board as game_board  # Use import alias to avoid class name conflict
from game.enums import Direction, MoveType

# --- PyTorch Imports ---
# These are needed for the neural network
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch not found. Agent will use a simple heuristic.")
    TORCH_AVAILABLE = False
# ---------------------

"""
This is an advanced agent that uses a Convolutional Neural Network (CNN) 
to evaluate the board state.
"""

# --- 1. Neural Network Definition ---

if TORCH_AVAILABLE:
    class SimpleNet(nn.Module):
        """
        A simple CNN to evaluate the board state.
        It takes two inputs:
        1. Spatial features (8x8 grids)
        2. Non-spatial features (a flat vector)
        """
        def __init__(self, spatial_channels=7, non_spatial_features=4):
            super(SimpleNet, self).__init__()
            
            # Convolutional layers for 8x8 spatial data
            self.conv1 = nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            
            # Calculate the flattened size after conv layers (8x8 grid remains 8x8)
            conv_out_size = 32 * 8 * 8
            
            # Fully connected layers
            self.fc1 = nn.Linear(conv_out_size + non_spatial_features, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1) # Output a single score

        def forward(self, x_spatial, x_non_spatial):
            # Process spatial data
            x = F.relu(self.conv1(x_spatial))
            x = F.relu(self.conv2(x))
            
            # Flatten conv output
            x = x.view(x.size(0), -1) 
            
            # Concatenate with non-spatial data
            combined = torch.cat((x, x_non_spatial), dim=1)
            
            # Process through fully connected layers
            combined = F.relu(self.fc1(combined))
            combined = F.relu(self.fc2(combined))
            
            # Output the final score
            # We use tanh to