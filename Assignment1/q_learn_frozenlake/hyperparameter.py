## Hyperparameters
N_EPISODE = 5000
MAX_STEPS_EPISODE = 100

ALPHA = 0.1       # learning rate
GAMMA = 0.99      # discount factor
EPSILON = 0.1     # exploration rate
DECAY_RATE = 0.99 # decay rate for epsilon

## Gymnasium Environment
REWARD_SCH = (1, -1, 0)