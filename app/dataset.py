import torch
import pandas as pd
import json
from torch.utils.data import Dataset
from data_preparation import prepare_training_pairs

class ChessDataset(Dataset):
    def __init__(self, dataset):
        """
        Custom PyTorch dataset for chess move prediction.

        Args:
            file_path (str): Path to the dataset file.
            file_type (str): "csv" or "json"
        """
        self.data = self.load_data(file_path, file_type)
        self.training_pairs = prepare_training_pairs(self.data)

    def load_data(self, file_path, file_type):
        """Loads chess data from CSV or JSON."""
        if file_type == "csv":
            df = pd.read_csv(file_path)
            return list(zip(df["FEN"], df["Move"]))
        elif file_type == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
            return [(entry["fen"], entry["move"]) for entry in data]
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'json'.")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.training_pairs)

    def __getitem__(self, id):
        """
        Returns a single sample from the dataset.

        Args:
            id: Index of the sample

        Returns:
            A tuple (fen_tensor, move_label)
        """
        return self.training_pairs[id]