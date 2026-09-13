import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Network(nn.Modules):
    def __init__(self):
        super(Network, self).__init__()
        self.layer_relu = nn.Sequentian(
            nn.linear(28*28, 512),
            nn.ReLU(),
            nn.linear(512, 512),
            nn.ReLU()
        )
    def forward(self, x):
        logic = self.layer_relu(x)
        return logic

model = Network()