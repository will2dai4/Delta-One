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