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
from IPython import display
import numpy as np

from logic_env import LogicEnv

print("finished importing")

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

class DeepQNetwork(nn.Module):
    
    # currently has an arbitrarily defined depth of 3, width of 64
    # changing the depth and width could be something worth looking into
    def __init__(self, n_observations, n_actions, lr):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        # the last layer is not activated, we want the raw output of the function before it is ReLU'd 
        return self.layer3(x)
    
class Agent():

    def run(self):
        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
            
        # turns on interactive mode for plotting software (what does that mean?)
        plt.ion()

        self.run_models()

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

        self.is_ipython = False
        # BATCH_SIZE is the number of transitions sampled from the ReplayMemory object.
        self.BATCH_SIZE = 128

        # GAMMA is the discount factor.
        # A low value means the system prioritizes immediate rewards,
        # a higher value proritizes future reward it can be confident in earning.
        self.GAMMA = 0.98

        # The algorithm used is the epsilon-greedy algorithm.
        # When the epsilon value is higher, the system is more likely to choose actions at random.
        # When it is lower, the system is more likely to choose actions based off the model.

        # EPS_START is the starting value of epsilon
        self.EPS_START = 0.95
        # EPS_END is the final value of epsilon
        self.EPS_END = 0.05
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.EPS_DECAY = 2000

        # TAU is the update rate of the target network
        self.TAU = 0.005
        # LR is the learning rate of the ``AdamW`` optimizer
        self.LR = 3e-4

        self.GRID_LENGTH = 4
        # Get the number of inputs
        self.n_observations = (self.GRID_LENGTH ** 2)
        # Define the number of outputs
        self.n_actions = (self.GRID_LENGTH ** 2) * 4 # see logic_env for descriptions on how actions are worked

        self.policy_net = DeepQNetwork(self.n_observations, self.n_actions, self.LR).to(self.device)
        self.target_net = DeepQNetwork(self.n_observations, self.n_actions, self.LR).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.memory = ReplayMemory(10000) # keep the last 10000 transitions for random sampling later
        self.episode_scores = []
        self.env = LogicEnv()

    def select_action(self, observation):
        self.steps_done += 1
        epsilon_limit = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

        if random.random() > epsilon_limit:
            with torch.no_grad():
                # exploit best known action
                # get the index of the maximum value, excluding invalid moves
                decision_outputs = self.policy_net(observation).tolist()[0]
                valid_moves = self.env.get_valid_moves()
                max_i = valid_moves[0]
                max_val = decision_outputs[max_i]

                for i, val in enumerate(decision_outputs):
                    if val > max_val and (i in valid_moves):
                        max_i = i
                        max_val = val

                return torch.tensor([[max_i]])
        else:
            # random action
            valid_moves = self.env.get_valid_moves()
            random_move = torch.tensor([[random.choice(valid_moves)]])
            return torch.tensor([[random_move]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        
        # do nothing if there aren't enough transition samples in the memory
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_scores(self, show_result=False):
        plt.figure(1)
        scores_t = torch.tensor(self.episode_scores, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(scores_t.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def run_models(self):
        # change these numbers to whatever value you want, might make it easier to do so later
        if torch.cuda.is_available():
            num_episodes = 90000
        else:
            num_episodes = 90000

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            recording = False
            file_name = None

            state, info = self.env.reset(record=recording, file_name=file_name)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, info = self.env.step(action)
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_scores.append(info)
                    #self.plot_scores()
                    self.print_running_avg(num=250)
                    self.print_running_avg(num=1000)
                    break

        print('Complete')
        self.plot_scores(show_result=True)
        plt.ioff()
        plt.show()

    def print_running_avg(self, num=100):
        if len(self.episode_scores) % num == 0 and len(self.episode_scores) != 0:
            window = self.episode_scores[len(self.episode_scores) - num : len(self.episode_scores)]
            running_avg = sum(window) // num
            print(f"Average of {len(self.episode_scores) - num} to {len(self.episode_scores)}: {running_avg}")
            

    def save_model(self):
        pass