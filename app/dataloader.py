import torch
from torch.utils.data import DataLoader
from dataset import ChessDataset

file_path = "chess_data.csv"
dataset = ChessDataset(file_path, file_type="csv")

data_loader = DataLoader(chess_dataset, batch_size=2, shuffle=True)

for batch in data_loader:
    fen_tensors, move_labels = batch
    print("Batch of FEN tensors:", fen_tensors.shape)
    print("Batch of move labels:", move_labels)
    break