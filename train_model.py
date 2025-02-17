import chess.pgn

def load_chess_data(file_path):
    # Loading the PGN file
    pgn = open(file_path)
    game = chess.pgn.read_game(pgn_file)

    board = game.board()
    positions = []

    for move in game.mainline_moves():
        board.push(move)
        positions.append(board.fen())

    return positions