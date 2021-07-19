from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as dataloader
import csv
import sys

from env import *
from cnn_model import *
from image_treatment import *
from image_gest import *
from benchmark_excel import *

file_list = [sys.argv[1] + f for f in listdir(sys.argv[1]) if ('.png' in f)]

for i in range(len(file_list)):
    image = Image.open(file_list[i])
    image.save(file_list[i][:-4] + ".jpeg")