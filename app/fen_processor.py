import torch

piece_map = {
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
    'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6,
    '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0
}

def fen_to_tensor(fen):
    '''
    Converts a FEN string to an 8x8 PyTorch tensor.
    '''

    board_state = fen.split()[0] # Get the board state from the FEN string
    tensor_board = torch.zeros((8, 8), dtype=torch.int8) # Initialize an 8x8 tensor

    row = 0
    col = 0

    for char in board_state:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            tensor_board[row][col] = piece_map[char]
            col += 1

    return tensor_board


def square_to_index(square):
    """
    Convert a square in algebraic notation (e.g., "e2") to a 0-indexed board position.
    """
    file = square[0]  # Letter part (e.g., 'e')
    rank = square[1]  # Number part (e.g., '2')
    
    # Convert file ('a' to 'h') to a number (0 to 7)
    file_index = ord(file) - ord('a')
    
    # Convert rank ('1' to '8') to a number (0 to 7), counting from the bottom
    rank_index = int(rank) - 1

    # Calculate the 0-indexed position on the board
    return rank_index * 8 + file_index


def move_to_label(move):
    """
    Convert a move in algebraic notation (e.g., "e2e4") to a unique numerical label in base 64.
    """
    from_square = move[:2]  # First two characters (e.g., 'e2')
    to_square = move[2:]    # Last two characters (e.g., 'e4')
    
    # Convert both squares to indices
    from_index = square_to_index(from_square)
    to_index = square_to_index(to_square)
    
    # Combine the two indices into a single label
    label = from_index * 64 + to_index

    return label