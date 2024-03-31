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

# named tuples allow for indexing by either name like a dictionary, or by index like a list
# should I need a terminal state to indicate whether or not the game is over?
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

class DeepQNetwork:
    
    # currently has an arbitrarily defined depth of 3, width of 128
    # changing the depth and width could be something worth looking into
    def __init__(self, n_observations, n_actions):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        return self.layer3(x)
    
class Agent():

    def __init__(self):
        self.steps_done = 0
        # BATCH_SIZE is the number of transitions sampled from the ReplayMemory object.
        self.BATCH_SIZE = 128

        # GAMMA is the discount factor.
        # A low value means the system prioritizes immediate rewards,
        # a higher value proritizes future reward it can be confident in earning.
        self.GAMMA = 0.99

        # The algorithm used is the epsilon-greedy algorithm.
        # When the epsilon value is higher, the system is more likely to choose actions at random.
        # When it is lower, the system is more likely to choose actions based off the model.

        # EPS_START is the starting value of epsilon
        self.EPS_START = 0.9
        # EPS_END is the final value of epsilon
        self.EPS_END = 0.05
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.EPS_DECAY = 1000

        # TAU is the update rate of the target network
        self.TAU = 0.005
        # LR is the learning rate of the ``AdamW`` optimizer
        self.LR = 1e-4

        self.GRID_LENGTH = 4
        # Get the number of inputs
        self.n_observations = (self.GRID_LENGTH ** 2)
        # Define the number of outputs
        self.n_actions = (self.GRID_LENGTH ** 2) * 2

        self.policy_net = DeepQNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DeepQNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        def select_action(state):
            #global steps_done
            # the "global" keyword defines a variable to the global scope, but it is already defined? is this needed?
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY) # the standalone backslash tells python to continue looking 
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

class DQNSystem:

    def run():
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

        