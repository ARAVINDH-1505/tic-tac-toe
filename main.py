import numpy as np
from dqn_agent import DQNAgent

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0=empty, 1=agent, -1=opponent

    def reset(self):
        self.board.fill(0)
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action, player):
        # Place player's mark
        self.board[action[0], action[1]] = player

        done, winner = self.check_done()
        reward = 0
        if done:
            if winner == 1:       # agent wins
                reward = 1
            elif winner == -1:    # opponent wins
                reward = -1
            else:                 # draw
                reward = 0
        return self.get_state(), reward, done

    def check_done(self):
        # Check rows, cols, diagonals
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return True, np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 3:
                return True, np.sign(sum(self.board[:, i]))
        diag1 = sum(self.board[i, i] for i in range(3))
        if abs(diag1) == 3:
            return True, np.sign(diag1)
        diag2 = sum(self.board[i, 2 - i] for i in range(3))
        if abs(diag2) == 3:
            return True, np.sign(diag2)
        # Draw if no empty spaces left
        if len(self.available_actions()) == 0:
            return True, 0
        return False, None

def opponent_move(env):
    # Simple opponent: choose random available action
    actions = env.available_actions()
    return actions[np.random.choice(len(actions))]

def train(num_episodes):
    env = TicTacToe()
    agent = DQNAgent()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            actions = env.available_actions()

            # Agent chooses action
            action = agent.act(state, actions)
            
            next_state, reward, done = env.step(action, player=1)  # Agent plays as 1

            if not done:
                # Opponent move
                opp_action = opponent_move(env)
                next_state, opp_reward, done = env.step(opp_action, player=-1)

                if done:
                    # If opponent wins or draw
                    if opp_reward == 1:
                        reward = -1  # penalty for agent losing
                    else:
                        reward = 0

            # Remember and learn
            agent.remember_and_learn(state, action, reward, next_state, done)

            state = next_state
            agent.step_count += 1

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} - Epsilon: {agent.epsilon:.3f}")

    # Save the model after training
    agent.save()

if __name__ == "__main__":
    train(10000)
