## Hyperparameters
N_EPISODE = 5000


GAMMA = 0.99           # discount factor (run 3)
EPSILON = 0.8       # exploration rate
DECAY_RATE = 0.998  # decay rate for epsilon
MIN_EPSILON = 0.005    # minimum exploration rate
MEMORY_CAP = 50000   # capacity of experience replay buffer
C = 500            # Every C steps, update the target network with the current network weights

LR = 0.001           # learning rate

