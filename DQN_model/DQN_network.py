import torch.nn as nn

class NeuralNetwork(nn.Module):

    # currently has an arbitrarily defined depth of 3, width of 128
    # changing the depth and width could be something worth looking into
    def __init__(self, n_observations, n_actions):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        return self.layer3(x)