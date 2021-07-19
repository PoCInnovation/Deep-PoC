import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool3d
import torch.utils.data as dataloader
import time
from env import *
from PIL import Image, ImageEnhance
from random import seed
from random import random

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(10, 10), padding=(3, 3)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(in_channels=4, out_channels=5, kernel_size=(6, 6), padding=(3, 4), stride=(1, 1)),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=5, out_channels=3, kernel_size=(3, 3), padding=(3, 3)),
            nn.MaxPool2d((1, 1)),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2)),
            nn.Flatten(1),
            nn.Linear(3588, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def backward(self, loss):
        loss.backward()