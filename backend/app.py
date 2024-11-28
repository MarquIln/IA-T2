from flask import Flask, request, jsonify
from flask_cors import CORS
from models.simulateGame import board_inputs, check_state
from models.algGenetico import GeneticAlgorithm
from models.minimax import Minimax
from copy import deepcopy
import threading
import numpy as np
from models.mlp import MLP

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, 
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response



board = ['b'] * 9  
nn = None
difficulty = "hard"  
nn_trained = False   
is_training = False

@app.route('/train', methods=['POST'])
def train_network():
    global is_training, nn, nn_trained

    if is_training:
        return jsonify({"status": "error", "message": "Training already in progress"}), 400

    def train():
        global is_training, nn, nn_trained
        is_training = True

        try:
            max_generations = 1000
            target_fitness = 0.99
            ga = GeneticAlgorithm()

            easy_percent = 25
            medium_percent = 25
            hard_percent = 50

            total_percent = easy_percent + medium_percent + hard_percent
            if total_percent != 100:
                raise ValueError("Difficulty percentages must add up to 100%.")

            easy_generations = (max_generations * easy_percent) // 100
            medium_generations = (max_generations * medium_percent) // 100
            hard_generations = max_generations - (easy_generations + medium_generations)

            difficulties = (['easy'] * easy_generations +
                            ['medium'] * medium_generations +
                            ['hard'] * hard_generations)

            best_global_fitness = 0
            best_global_nn = None
            best_generation = None

            for generation in range(max_generations):
                difficulty = difficulties[min(generation, len(difficulties) - 1)]

                ga.evolve_population(difficulty)

                for _ in range(10):
                    simulated_board = ['b'] * 9
                    minimax = Minimax(simulated_board, difficulty)
                    minimax.move()
                    best_nn = ga.population[0]
                    ga.fitness(best_nn, difficulty)

                best_nn = ga.population[0]
                best_fitness = ga.fitness(best_nn, difficulty)

                if best_fitness > best_global_fitness:
                    best_global_fitness = best_fitness
                    best_global_nn = deepcopy(best_nn)
                    best_generation = generation + 1

                print(f"Generation {generation+1}, Best Fitness of Generation: {best_fitness:.2f}, "
                      f"Global Best Fitness: {best_global_fitness:.2f} (Generation {best_generation}), "
                      f"Difficulty: {difficulty}")

                if best_global_fitness >= target_fitness:
                    print(f"Target fitness achieved in generation {generation+1}")
                    break

            np.savez('./weights_network.npz',
                     weights_input_hidden=best_global_nn.weights_input_hidden,
                     weights_hidden_output=best_global_nn.weights_hidden_output,
                     bias_hidden=best_global_nn.bias_hidden,
                     bias_output=best_global_nn.bias_output)

            load_trained_network()

        except Exception as e:
            print(f"Error during training: {e}")

        finally:
            is_training = False

    threading.Thread(target=train).start()
    return jsonify({"status": "success", "message": "Training started in the background"}), 202


def load_trained_network():
    global nn, nn_trained
    try:
        data = np.load('./assets/best_network_weights.npz')
        nn = MLP()
        nn.weights_input_hidden = data['weights_input_hidden']
        nn.weights_hidden_output = data['weights_hidden_output']
        nn.bias_hidden = data['bias_hidden']
        nn.bias_output = data['bias_output']
        nn_trained = True
    except FileNotFoundError:
        nn = None
        nn_trained = False


@app.route('/move', methods=['POST'])
def make_move():
    global board, nn

    data = request.json
    position = data.get('position')
    mode = data.get('mode', 'minimax')

    if not (0 <= position < 9) or board[position] != 'b':
        return jsonify({"status": "error", "message": "Invalid move"}), 400

    board[position] = 'x'
    state = check_state(board)

    if state != "Continue":
        return jsonify({"status": "game_over", "state": state, "board": board})

    if mode == "minimax":
        minimax = Minimax(board, difficulty)
        move = minimax.move()
    elif mode == "neural" and nn_trained:
        inputs = board_inputs(board)
        outputs = nn.forward(inputs)
        available_indices = [i for i, x in enumerate(board) if x == 'b']
        move = max(available_indices, key=lambda idx: outputs[idx])
    else:
        return jsonify({"status": "error", "message": "Invalid mode or neural network not trained"}), 400

    board[move] = 'o'
    state = check_state(board)

    return jsonify({"status": "success", "state": state, "board": board})

@app.route('/reset', methods=['POST'])
def reset_game():
    global board
    board = ['b'] * 9
    return jsonify({"status": "success", "board": board})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
