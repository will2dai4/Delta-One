import torch
import torch.nn as nn
import torch.optim as optim

class ChessMovePredictor(nn.Module):
    def __init__(self, input_size=64, hidden_size=256, output_size=4096):
        """
        Neural network for chess move prediction.

        Args:
            input-size (int):Number of input features (8x8 board = 64 input features)
            hidden_size (int): Number of neurons in hidden layers.
            output_size (int): Number of possible moves (adjustable)
        """
        super(ChessMovePredictor, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftware(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Format input
        x = self.relu(self.fc1(x)) # Pass through first hidden layer
        x = self.relu(self.fc2(x)) # Pass through second hidden layer
        x = self.softmax(self.fc3(x)) # Predict move probabilities
        return x