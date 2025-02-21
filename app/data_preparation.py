from fen_processor import fen_to_tensor, move_to_label

def prepare_training_pairs(dataset):
    """
    Prepare training pairs of (FEN tensor, move label).
    
    Args:
        dataset: List of tuples [(FEN, move), ...]

    Returns:
        List of (tensor, label) pairs for training.
    """
    training_pairs = []

    for fen, move in dataset:
        try:
            # Convert FEN to tensor
            fen_tensor = fen_to_tensor(fen)
            
            # Encode the move as a label
            move_label = move_to_label(move)
            
            # Append the (tensor, label) pair to the list
            training_pairs.append((fen_tensor, move_label))
        except Exception as e:
            print(f"Error processing FEN {fen} or move {move}: {e}")
    
    return training_pairs
