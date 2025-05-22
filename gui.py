import tkinter as tk
from environment import TicTacToe
from dqn_agent import DQNAgent

class TicTacToeGUI:
    def __init__(self):
        self.game = TicTacToe()
        self.agent = DQNAgent()
        try:
            self.agent.load()
            print("Model loaded.")
        except:
            print("No saved model found, starting fresh.")

        self.window = tk.Tk()
        self.window.title("Tic Tac Toe with DQN")

        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_board()

        self.player_turn = 1  # Human is 'X' = 1
        self.game.reset()
        self.update_gui()

    def create_board(self):
        for i in range(3):
            for j in range(3):
                btn = tk.Button(self.window, text="", font=('normal', 40), width=5, height=2,
                                command=lambda i=i, j=j: self.human_move(i, j))
                btn.grid(row=i, column=j)
                self.buttons[i][j] = btn

    def human_move(self, i, j):
        if self.game.done or self.game.board[i, j] != 0:
            return

        # Human plays as 1 (X)
        self.game.step((i, j), 1)
        self.update_gui()

        if self.game.done:
            self.show_result()
            return

        # Agent turn after short delay
        self.window.after(500, self.agent_move)

    def agent_move(self):
        if self.game.done:
            self.show_result()
            return

        state = self.game.get_state()
        actions = self.game.available_actions()
        action = self.agent.act(state, actions)

        next_state, reward, done = self.game.step(action, -1)  # Agent plays -1 (O)
        self.update_gui()

        # Agent learns from this move
        self.agent.remember_and_learn(state, action, reward, next_state, done)
        self.agent.save()

        if done:
            self.show_result()

    def update_gui(self):
        for i in range(3):
            for j in range(3):
                if self.game.board[i, j] == 1:
                    self.buttons[i][j]["text"] = "X"
                    self.buttons[i][j]["state"] = "disabled"
                elif self.game.board[i, j] == -1:
                    self.buttons[i][j]["text"] = "O"
                    self.buttons[i][j]["state"] = "disabled"
                else:
                    self.buttons[i][j]["text"] = ""
                    self.buttons[i][j]["state"] = "normal" if not self.game.done else "disabled"

    def show_result(self):
        if self.game.winner == 1:
            result_text = "You win!"
        elif self.game.winner == -1:
            result_text = "Agent wins!"
        else:
            result_text = "It's a draw!"

        result_label = tk.Label(self.window, text=result_text, font=('normal', 30))
        result_label.grid(row=3, column=0, columnspan=3)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    gui = TicTacToeGUI()
    gui.run()
