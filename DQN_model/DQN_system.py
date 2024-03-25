import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQN_network import NeuralNetwork

class DQNSystem:

    def run():
        
        # named tuples allow for indexing by either name like a dictionary, or by index like a list
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        class ReplayMemory(object):
            # stores transitions in a deque
            # deque (pronounced deck) are double-ended queues

            def __init__(self, capacity):
                self.memory = deque([], maxlen=capacity)

            def push(self, *args):
                # adds a transition onto the deque
                self.memory.append(Transition(*args))

            def sample(self, batch_size):
                # returns a random transition from the memory
                return random.sample(self.memory, batch_size)

            def __len__(self):
                return len(self.memory)

        # set up matplotlib
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            print("you need to install Ipython :(")
            # from IPython import display
            
        # turns on interactive mode for plotting software (what does that mean?)
        plt.ion()

        # cuda allows operations to run on the GPU
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # BATCH_SIZE is the number of transitions sampled from the ReplayMemory object.
        BATCH_SIZE = 128

        # GAMMA is the discount factor.
        # A low value means the system prioritizes immediate rewards,
        # a higher value proritizes future reward it can be confident in earning.
        GAMMA = 0.99

        # The algorithm used is the epsilon-greedy algorithm.
        # When the epsilon value is higher, the system is more likely to choose actions at random.
        # When it is lower, the system is more likely to choose actions based off the model.

        # EPS_START is the starting value of epsilon
        EPS_START = 0.9
        # EPS_END is the final value of epsilon
        EPS_END = 0.05
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        EPS_DECAY = 1000

        # TAU is the update rate of the target network
        TAU = 0.005
        # LR is the learning rate of the ``AdamW`` optimizer
        LR = 1e-4

        GRID_LENGTH = 4
        # Get the number of inputs
        n_observations = (GRID_LENGTH ** 2)
        # Define the number of outputs
        n_actions = (GRID_LENGTH ** 2) * 2

        policy_net = NeuralNetwork(n_observations, n_actions).to(device)
        target_net = NeuralNetwork(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        steps_done = 0

        def select_action(state):
            global steps_done # the "global" keyword defines a variable to the global scope, but it is already defined? is this needed?
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY) # the standalone backslash tells python to continue looking 
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)