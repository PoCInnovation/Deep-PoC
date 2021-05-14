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
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 10, 10), padding=(1, 3, 3)),
            nn.MaxPool3d((1, 2, 1)),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 6, 6), padding=(1, 3, 4), stride=(2, 1, 1)),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), padding=(1, 3, 3)),
            nn.MaxPool3d((2, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2, 2, 2)),
            nn.Flatten(2),
            nn.Linear(2392, 512),
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