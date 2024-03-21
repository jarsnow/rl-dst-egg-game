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
        
        # named tuples allow for indexing by 
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        class ReplayMemory(object):

            def __init__(self, capacity):
                self.memory = deque([], maxlen=capacity)

            def push(self, *args):
                """Save a transition"""
                self.memory.append(Transition(*args))

            def sample(self, batch_size):
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