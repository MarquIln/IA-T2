import random
import numpy as np
from models.minimax import Minimax

def simulate_game(neural_network, difficulty, verbose=True, game_number=1):
    board = generate_random_board()
    current_player = 'nn'
    move_count = 0

    while True:
        if current_player == 'nn':
            move = play_nn(neural_network, board)
            if move is None or board[move] != 'b':
                if verbose:
                    print(f"Invalid move by neural network at move {move_count}.")
                return 0 
            board[move] = 'nn'
            state = check_state(board)
            if state != "Continue":
                if verbose:
                    print(f"Game over (NN). Final state: {state}.")
                return game_result(state)
            current_player = 'minimax'
        else:
            minimax = Minimax(board.copy(), difficulty)
            move = minimax.move()
            if move is None or board[move] != 'b':
                if verbose:
                    print(f"Invalid move by neural network at game {game_number}. Penalizing.")
                return -0.5
            board[move] = 'mm'
            state = check_state(board)
            if state != "Continue":
                if verbose:
                    print(f"Game over (MM). Final state: {state}.")
                return game_result(state)
            current_player = 'nn'
        move_count += 1

def generate_random_board():
    board = ['b'] * 9
    move_count = np.random.randint(2, 5)
    for _ in range(move_count):
        empty_indices = [i for i, x in enumerate(board) if x == 'b']
        if not empty_indices:
            break
        board[random.choice(empty_indices)] = random.choice(['nn', 'mm'])
    return board

def play_nn(neural_network, board):
    inputs = board_inputs(board)
    outputs = neural_network.forward(inputs)
    available_indices = [i for i, x in enumerate(board) if x == 'b']
    if not available_indices:
        return None
    output_indices = [(i, outputs[i]) for i in available_indices]
    output_indices.sort(key=lambda x: x[1], reverse=True)
    return output_indices[0][0]

def board_inputs(board):
    mapping = {'b': 0, 'nn': 1, 'mm': -1}
    return np.array([mapping.get(x, 0) for x in board])

def check_state(board):
    combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                    (0, 3, 6), (1, 4, 7), (2, 5, 8),
                    (0, 4, 8), (2, 4, 6)]
    for a, b, c in combinations:
        if board[a] == board[b] == board[c] != 'b':
            return f"Player {board[a].upper()} wins"
    if 'b' not in board:
        return "Draw"
    return "Continue"

def game_result(state):
    if state == "Player NN wins":
        return 1
    elif state == "Player MM wins":
        return -1
    elif state == "Draw":
        return -1
    else:
        return 0 

def print_final_board(board):
    icon_map = {'b': '     ', 'nn': '  NN ', 'mm': '  MM ', 'invalid': ' Inv '}
    display_board = [icon_map.get(x, '     ') for x in board]
    print(f"{display_board[0]}|{display_board[1]}|{display_board[2]}")
    print("-----+-----+-----")
    print(f"{display_board[3]}|{display_board[4]}|{display_board[5]}")
    print("-----+-----+-----")
    print(f"{display_board[6]}|{display_board[7]}|{display_board[8]}")
    print()