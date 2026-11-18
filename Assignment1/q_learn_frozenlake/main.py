import gymnasium as gym
from hyperparameter import *
from q_agent import Q_Agent
from plots import plot_q_tables, plot_reward_per_episode, plot_avg_steps_to_goal
import numpy as np

# Environment
env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="4x4",
        is_slippery=False,
        success_rate=1.0/3.0,
        reward_schedule=REWARD_SCH,
        max_episode_steps=MAX_STEPS_EPISODE,
 )
env.reset()

# Agent
agent = Q_Agent(
    n_actions=env.action_space.n,
    n_states=env.observation_space.n,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon_0=EPSILON,
    decay_rate=DECAY_RATE
)

# Logging
SNAPSHOT_EPISODES = {500, 2000, N_EPISODE}
q_snapshots = {}
episode_rewards = []
episode_steps = []

# Q-learning loop
for episode in range(N_EPISODE):
    print(f"Episode {episode + 1}/{N_EPISODE}")
    s, _ = env.reset()
    done = False
    total_reward = 0
    n_steps = 0
    reached_goal = False
    while not done:
        a = agent.choose_action(s, episode)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total_reward += r
        n_steps += 1
        if not truncated:
            target = r
            if not terminated:
                target += agent.gamma * np.max(agent.q_table[s_next])
            agent.q_table[s, a] = (1 - agent.alpha) * agent.q_table[s, a] \
                                    + agent.alpha * target
            s = s_next
        reached_goal = terminated and r == REWARD_SCH[0]

    episode_rewards.append(total_reward)
    episode_steps.append(n_steps if reached_goal else MAX_STEPS_EPISODE)
    if episode + 1 in SNAPSHOT_EPISODES:
        q_snapshots[episode + 1] = agent.q_table.copy()

# Plots
plot_q_tables(q_snapshots)
plot_reward_per_episode(episode_rewards)
plot_avg_steps_to_goal(episode_steps, window=100)
