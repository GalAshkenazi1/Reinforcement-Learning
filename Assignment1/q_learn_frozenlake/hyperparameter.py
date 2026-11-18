## Hyperparameters
N_EPISODE = 5000
MAX_STEPS_EPISODE = 100

ALPHA = 0.2       # learning rate
GAMMA = 1     # discount factor (run 3)
EPSILON = 0.8     # exploration rate
DECAY_RATE = 0.998 # decay rate for epsilon

## Gymnasium Environment
REWARD_SCH = (10, -10, 0) # (reaching goal, falling into hole, stepping on frozen tile)