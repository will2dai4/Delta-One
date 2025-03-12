import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessMovePredictor(nn.Module):
    def __init__(self, output_size=4096):
        """
        CNN-based chess move prediction model.

        Args:
            output_size (int): Number of possible moves (default to 4096)
        """
        super(ChessMovePredictor, self).__init__()

        # Convolutional layers (spatial learning)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, padding=1)  # 8 filters
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, padding=1) # 16 filter

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 256)  # Flattened 16x8x8 board
        self.fc2 = nn.Linear(256, output_size) # Predict move probabilities

        # Reduces overfitting
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = x.view(-1, 1, 8, 8) 
        x = F.relu(self.conv1(x))  # Applying first convolution + ReLU
        x = F.relu(self.conv2(x))  # Applying second convolution + ReLU
        x = x.view(x.size(0), -1)  # Flattening the board
        x = F.relu(self.fc1(x))    # First fully connected layer
        x = self.dropout(x)        # Applying the dropout
        x = self.fc2(x)            # Final output layer (move probabilities)
        return x