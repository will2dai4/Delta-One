from flask import Flask, request, jsonify
import torch

from fen_processor import fen_to_tensor

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_move():
    """
    Predict the next move in a given position for a chess game.

    Request:
    {
        "board": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }
    Response:
    {
        "next_move": "e2e4"
    }
    """
    data = request.json
    fen = data["fen"]
    
    # Convert FEN to tensor
    board_tensor = fen_to_tensor(fen)
    
    # Predict the best move
    with torch.no_grad():
        output = model(board_tensor)
        predicted_move = torch.argmax(output).item()
    
    return jsonify({"next_move": predicted_move})

# Testing endpoint
@app.route("/fen", methods=["POST"])
def process_fen():
    data = request.get_json() 
    fen = data.get("fen", "No FEN provided")
    return jsonify({"received_fen": fen})


if __name__ == '__main__':
    app.run(debug=True)