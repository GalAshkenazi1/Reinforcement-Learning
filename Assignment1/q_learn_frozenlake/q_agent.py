import numpy as np
import matplotlib.pyplot as plt

class Q_Agent():
    def __init__(self, n_actions, n_states, alpha, gamma, epsilon_0, decay_rate = 1):
        # Default to no-decay
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_0 = epsilon_0
        self.decay_rate = decay_rate
        self.q_table = np.zeros((n_states, n_actions))

        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= self.gamma <= 1, "Gamma must be between 0 and 1"
        assert 0 <= self.epsilon_0 <= 1, "Epsilon must be between 0 and 1"
        assert 0 <= self.decay_rate <= 1, "Decay rate must be between 0 and 1"

    def choose_action(self, state, episode):
        epsilon = self.epsilon_0 * (self.decay_rate ** episode)
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def plot_q_table(self, iteration):
        f =plt.imshow(self.q_table, cmap='hot')
        f.colorbar()
        f.title("Q-Table Heatmap")
        f.xlabel("Actions")
        f.ylabel("States")
        return f