import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 9)  # Q values for each board position

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork().to(self.device)
        self.target_model = DQNNetwork().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_freq = 1000  # update target net every 1000 steps
        self.step_count = 0

    def action_to_index(self, action):
        row, col = action
        return row * 3 + col

    def get_state_tensor(self, state):
        # state is assumed to be 3x3, flatten to (1,9)
        state = np.array(state).reshape(1, -1)
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def act(self, state, available_actions):
        self.model.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1,9)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()

        # filter only available actions by their indices
        q_values_filtered = {action: q_values[self.action_to_index(action)] for action in available_actions}

        # choose the best action with max Q-value
        best_action = max(q_values_filtered, key=q_values_filtered.get)
        return best_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def remember_and_learn(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.memory) >= self.batch_size:
            self.learn()

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.step_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat([self.get_state_tensor(s) for s in states])
        next_states = torch.cat([self.get_state_tensor(s) for s in next_states])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)

        target_q_values = q_values.clone()
        for idx in range(self.batch_size):
            action = actions[idx]
            action_index = self.action_to_index(action)
            max_next_q = next_q_values[idx].max()
            target = rewards[idx] + (1 - dones[idx]) * self.gamma * max_next_q
            target_q_values[idx, action_index] = target

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="dqn_model.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path="dqn_model.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
