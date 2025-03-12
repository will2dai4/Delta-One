import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessMovePredictor

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Device Configuration
# Change "cpu" to "cuda" if available
device = torch.devive("cpu")

# Loading the dataset
dataset = ChessDataset("data/chess_data.csv")
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initializing the model
model = ChessMovePredictor().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.paramters(), lr=LEARNING_RATE)


# Training Loop
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch in data_loader:
        fen_tensors, move_labels = batch
        fen_tensors, move_labals = fen_tensors.to(device), move_labels.to(device)

        # Forward pass: compute predictions
        outputs = model(fen_tensors)

        # Compute loss
        loss = criterion(outputs, move_labels)

        # Backward pass: computing gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_lost += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "chess_model.pth")
print("Training complete. Model saved.")
