# environment.py
import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3,3), dtype=int)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        # Flatten the board to a 9-length vector
        return self.board.flatten().astype(float)

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i,j] == 0]

    def step(self, action, player):
        if self.done:
            raise Exception("Game is over, reset to start a new game.")

        i, j = action
        if self.board[i, j] != 0:
            raise ValueError("Invalid move! Position already taken.")

        self.board[i, j] = player
        self.done, self.winner = self.check_game_over()

        reward = 0
        if self.done:
            if self.winner == player:
                reward = 1    # Win
            elif self.winner == 0:
                reward = 0.5  # Draw
            else:
                reward = -1   # Loss
        return self.get_state(), reward, self.done

    def check_game_over(self):
        # Check rows, columns, diagonals
        lines = []

        lines.extend([self.board[i, :] for i in range(3)])  # rows
        lines.extend([self.board[:, j] for j in range(3)])  # cols
        lines.append(np.diag(self.board))                   # main diag
        lines.append(np.diag(np.fliplr(self.board)))        # anti diag

        for line in lines:
            if np.all(line == 1):
                return True, 1
            if np.all(line == -1):
                return True, -1

        if np.all(self.board != 0):
            return True, 0  # Draw

        return False, None
