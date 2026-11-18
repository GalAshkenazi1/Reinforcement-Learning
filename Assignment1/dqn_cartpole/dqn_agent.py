import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import collections
import gymnasium as gym

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon_0=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 n_hidden=3, lr=0.001, memory_cap=2000, minibatch_size=64, c=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_0 = epsilon_0
        self.epsilon_min = epsilon_min  # minimum exploration rate
        self.epsilon_decay = epsilon_decay

        # NN
        self.net = nn.Sequential( # 3 layer
            nn.Linear(state_size, 24),
            nn.ReLU(),
            *[layer for _ in range(n_hidden - 1) for layer in (nn.Linear(24, 24), nn.ReLU())],
            nn.Linear(24, action_size)
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Target network holds the old weights theta_old
        self.target_net = copy.deepcopy(self.net)


        # Experience replay buffer
        self.D = collections.deque(maxlen=memory_cap)
        self.minibatch_size = minibatch_size

        self.c = c

    def sample_batch(self):
        return random.sample(self.D, self.minibatch_size)

    def sample_action(self, state, episode):
        epsilon = max(self.epsilon_min, self.epsilon_0 * (self.epsilon_decay ** episode))
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            with torch.no_grad():
                return self.net(torch.as_tensor(state, dtype=torch.float32)).argmax().item()  # Exploit

    def train_agent(self, n_episodes, env: gym.Env):
        step_counter = 0  # global, so the target net syncs every c steps across episodes
        for episode in range(n_episodes):
            s = env.reset()[0]
            done = False
            while not done:
                step_counter += 1
                a = self.sample_action(s, episode)
                s_next, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated  # CartPole-v1 truncates at 500 steps
                self.D.append((s, a, r, s_next, terminated))
                s = s_next

                if len(self.D) < self.minibatch_size:
                    continue

                batch = self.sample_batch()
                states, actions, rewards, next_states, terms = zip(*batch)
                states =      torch.as_tensor(np.array(states), dtype=torch.float32)
                actions =     torch.as_tensor(actions, dtype=torch.int64)
                rewards =     torch.as_tensor(rewards, dtype=torch.float32)
                next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
                terms =       torch.as_tensor(terms, dtype=torch.float32)

                # Q(s, a) with current weights theta
                q_curr = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # targets with old weights theta_old (no gradient)
                with torch.no_grad():
                    q_old = self.target_net(next_states).max(dim=1).values
                    y = rewards + self.gamma * q_old * (1 - terms)

                loss = nn.MSELoss()(q_curr, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step_counter % self.c == 0:
                    self.target_net.load_state_dict(self.net.state_dict())  # theta_old <- theta


    def test_agent(self, env: gym.Env):

        env.render_mode = "human"
        s = env.reset()[0]
        env.render()

        terminated = False
        total_reward = 0
        while not terminated:
            with torch.no_grad():
                a = self.net(torch.as_tensor(s, dtype=torch.float32)).argmax().item()
            s_next, r, terminated, _, _ = env.step(a)
            total_reward += r
            s = s_next

        env.close()