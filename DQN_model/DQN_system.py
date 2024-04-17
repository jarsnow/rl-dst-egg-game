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

import numpy as np

from logic_env import LogicEnv

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
    def __init__(self, n_observations, n_actions, lr):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        # the last layer is not activated, we want the raw output of the function before it is ReLU'd 
        return self.layer3(x)
    
class Agent():

    def run():
        # set up matplotlib
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            print("you need to install Ipython :(")
            # from IPython import display
            
        # turns on interactive mode for plotting software (what does that mean?)
        plt.ion()

    def __init__(self):
        # indicates the amount of times an action has been selected
        self.steps_done = 0

        # self.env = 

        self.device = (
            # cuda allows operations to run on the GPU
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

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

        self.policy_net = DeepQNetwork(self.n_observations, self.n_actions, self.LR).to(self.device)
        self.target_net = DeepQNetwork(self.n_observations, self.n_actions, self.LR).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(10000)

        self.env = LogicEnv()

    def select_action(self, observation):
        epsilon = None
        if random.random() > epsilon:
            # exploit the currently best known solution
            state = torch.tensor([observation]).to(self.device)
            actions = self.policy_net.forward(state)
            chosen_action = torch.argmax(actions).item()
        else:
            # explore a random action
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
        self.steps_done += 1
        
    def store_transition(self, state, action, next_state, reward):
        new_transition = Transition(state, action, next_state, reward)
        self.memory.push(new_transition)

    def optimize_model(self):
        
        # do nothing if there aren't enough transition samples in the memory
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))