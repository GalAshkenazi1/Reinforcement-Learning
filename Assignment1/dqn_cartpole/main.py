import gymnasium as gym
from dqn_agent import DQNAgent
from hyperparameter import *
import numpy as np

# Environment
env = gym.make("CartPole-v1")
env.reset()

# Agent
agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
    gamma=GAMMA,
    epsilon_0=EPSILON,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    lr=LR,
    memory_cap=2000
)

agent.train_agent(n_episodes=N_EPISODE, env=env)

agent.test_agent(env=env)
