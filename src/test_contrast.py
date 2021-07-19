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

for i, elmnt in enumerate(sys.argv):
    if (i == 0):
        continue
    image = Image.open(elmnt)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(10)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(10)
    image.save("test_" + elmnt.split("/")[-1].split(".")[0] + ".jpeg")
