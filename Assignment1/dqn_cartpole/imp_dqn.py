import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import collections
import gymnasium as gym


torch.set_num_threads(1)

class ImprovedDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon_0=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 n_hidden=3, lr=0.001, memory_cap=2000, minibatch_size=64, c=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_0 = epsilon_0
        self.epsilon_min = epsilon_min  # minimum exploration rate
        self.epsilon_decay = epsilon_decay

        # NN
        self.netA = nn.Sequential( # 3 layer
            nn.Linear(state_size, 24),
            nn.ReLU(),
            *[layer for _ in range(n_hidden - 1) for layer in (nn.Linear(24, 24), nn.ReLU())],
            nn.Linear(24, action_size)
        )
        self.netB = nn.Sequential( # 3 layer
            nn.Linear(state_size, 24),
            nn.ReLU(),
            *[layer for _ in range(n_hidden - 1) for layer in (nn.Linear(24, 24), nn.ReLU())],
            nn.Linear(24, action_size)
        )
        self.optimizerA = optim.Adam(self.netA.parameters(), lr=lr)
        self.optimizerB = optim.Adam(self.netB.parameters(), lr=lr)

        # Target network holds the old weights theta_old
        self.target_netA = copy.deepcopy(self.netA)
        self.target_netB = copy.deepcopy(self.netB)
        self.lossfn =  nn.MSELoss()


        # Experience replay buffer
        self.D = collections.deque(maxlen=memory_cap)
        self.minibatch_size = minibatch_size

        self.c = c

        # Training history (for plotting)
        self.losses = []           # loss of every gradient step
        self.episode_rewards = []  # total reward of every training episode

    def sample_batch(self):
        return random.sample(self.D, self.minibatch_size)

    def sample_action(self, state, episode):
        epsilon = max(self.epsilon_min, self.epsilon_0 * (self.epsilon_decay ** episode))
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            with torch.no_grad():
                Q_A = self.netA(torch.as_tensor(state, dtype=torch.float32))
                Q_B = self.netB(torch.as_tensor(state, dtype=torch.float32))
                return (Q_A + Q_B).argmax().item()  # Exploit


    def train_agent(self, n_episodes, env: gym.Env):
        step_counter = 0  # global, so the target net syncs every c steps across episodes
        for episode in range(n_episodes):
            s = env.reset()[0]
            done = False
            episode_reward = 0
            if episode % 100 ==0:
                print(f"Episode: {episode}/{n_episodes}")
            while not done:
                step_counter += 1
                a = self.sample_action(s, episode)
                s_next, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated  # CartPole-v1 truncates at 500 steps
                episode_reward += r
                if done:
                    self.episode_rewards.append(episode_reward)
                self.D.append((s, a, r, s_next, terminated))
                s = s_next

                if len(self.D) < self.minibatch_size or step_counter % 4 != 0:
                    continue  # train only every 4 env steps

                batch = self.sample_batch()
                states, actions, rewards, next_states, terms = zip(*batch)
                states =      torch.as_tensor(np.array(states), dtype=torch.float32)
                actions =     torch.as_tensor(actions, dtype=torch.int64)
                rewards =     torch.as_tensor(rewards, dtype=torch.float32)
                next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
                terms =       torch.as_tensor(terms, dtype=torch.float32)

                update_A = np.random.rand() < 0.5  # randomly choose which network to update

                if update_A:
                    q_curr = self.netA(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                else:
                    q_curr = self.netB(states).gather(1, actions.unsqueeze(1)).squeeze(1)


                # targets with old weights theta_old (no gradient)
                with torch.no_grad():
                    if update_A:
                        a_next = self.netA(next_states).argmax(dim=1)
                        q_comp_old = self.target_netB(next_states).gather(1, a_next.unsqueeze(1)).squeeze(1)

                    else:
                        a_next = self.netB(next_states).argmax(dim=1)
                        q_comp_old = self.target_netA(next_states).gather(1, a_next.unsqueeze(1)).squeeze(1)

                    # y is r+gamma * max_a' Q_comp(s', a'; theta_old) if not terminal, else r
                    y = rewards + self.gamma * q_comp_old * (1 - terms)

                loss = self.lossfn(q_curr, y)
                self.losses.append(loss.item())

                if update_A:
                    self.optimizerA.zero_grad()
                    loss.backward()
                    self.optimizerA.step()
                else:
                    self.optimizerB.zero_grad()
                    loss.backward()
                    self.optimizerB.step()

                if step_counter % self.c == 0:
                    self.target_netA.load_state_dict(self.netA.state_dict())  # theta_old <- theta
                    self.target_netB.load_state_dict(self.netB.state_dict())  # theta_old <- theta


    def episodes_to_solve(self, threshold=475.0, window=100):
        """Number of training episodes until the average reward over `window`
        consecutive episodes first reaches `threshold`. None if never reached."""
        rewards = np.asarray(self.episode_rewards, dtype=float)
        if len(rewards) < window:
            return None
        avgs = np.convolve(rewards, np.ones(window) / window, mode="valid")
        hits = np.flatnonzero(avgs >= threshold)
        # avgs[i] is the mean of episodes i+1 .. i+window (1-based), so it ends at episode i+window
        return int(hits[0]) + window if len(hits) else None


    def _greedy_action(self, s):
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32)
            return (self.netA(s) + self.netB(s)).argmax().item()

    def _run_episode(self, env):
        s = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            s, r, terminated, truncated, _ = env.step(self._greedy_action(s))
            done = terminated or truncated  # CartPole-v1 truncates at 500 steps
            total_reward += r
        return total_reward

    def evaluate(self, env: gym.Env, n_episodes=10):
        """Headless greedy evaluation of the final policy. Prints a summary line, returns the rewards."""
        eval_env = gym.make(env.spec.id)
        rewards = [self._run_episode(eval_env) for _ in range(n_episodes)]
        eval_env.close()
        print(f"Test over {n_episodes} greedy episodes: mean {np.mean(rewards):.1f} "
              f"+- {np.std(rewards):.1f} (min {np.min(rewards):.0f}, max {np.max(rewards):.0f})")
        return rewards

    def test_agent(self, env: gym.Env, n_episodes=10):
        """Runs the final policy greedily (no exploration) for `n_episodes` episodes.
        The first episode is rendered; the rest run without a window. Returns the rewards."""
        # render_mode can't be changed on an existing env, so build new envs
        env.close()
        render_env = gym.make(env.spec.id, render_mode="human")
        rewards = [self._run_episode(render_env)]
        print(f"Test episode 1/{n_episodes}: {rewards[-1]}")
        render_env.close()

        quiet_env = gym.make(env.spec.id)
        for i in range(2, n_episodes + 1):
            rewards.append(self._run_episode(quiet_env))
            print(f"Test episode {i}/{n_episodes}: {rewards[-1]}")
        quiet_env.close()

        print(f"Test over {n_episodes} greedy episodes: mean {np.mean(rewards):.1f} "
              f"+- {np.std(rewards):.1f} (min {np.min(rewards):.0f}, max {np.max(rewards):.0f})")
        return rewards