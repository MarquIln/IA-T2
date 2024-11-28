import random

class Minimax:
    def __init__(self, board, difficulty):
        self.board = board
        self.difficulty = difficulty

    def move(self):
        if self.difficulty == 'easy':
            if random.random() < 0.25:
                return self.minimax_move()
            else:
                return self.random_move()
        elif self.difficulty == 'medium':
            if random.random() < 0.5:
                return self.minimax_move()
            else:
                return self.random_move()
        else:
            return self.minimax_move()

    def random_move(self):
        available_indices = [i for i, x in enumerate(self.board) if x == 'b']
        return random.choice(available_indices) if available_indices else None

    def minimax_move(self):
        best_score = -float('inf')
        best_move = None
        for i in range(9):
            if self.board[i] == 'b':
                self.board[i] = 'o'
                score = self.minimax(False)
                self.board[i] = 'b'
                if score > best_score:
                    best_score = score
                    best_move = i
        return best_move

    def minimax(self, is_maximizing):
        state = self.check_state()
        if state == "Player O wins":
            return 1
        elif state == "Player X wins":
            return -1
        elif state == "Draw":
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for i in range(9):
                if self.board[i] == 'b':
                    self.board[i] = 'o'
                    score = self.minimax(False)
                    self.board[i] = 'b'
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if self.board[i] == 'b':
                    self.board[i] = 'x'
                    score = self.minimax(True)
                    self.board[i] = 'b'
                    best_score = min(score, best_score)
            return best_score

    def check_state(self):
        combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)           
        ]
        for a, b, c in combinations:
            if self.board[a] == self.board[b] == self.board[c] != 'b':
                return f"Player {self.board[a].upper()} wins"
        if 'b' not in self.board:
            return "Draw"
        return "Continue"