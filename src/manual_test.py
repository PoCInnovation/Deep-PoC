import sys
import torch

from env import *
from cnn_model import *
from image_treatment import *
from image_gest import *
from benchmark_excel import *

img_treat = image_treatment()
cnn = CNN()
cnn.load_state_dict(torch.load(sys.argv[1]))

img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness = 2,1,1,1
for i, arg in enumerate(sys.argv):
    if (i <= 1):
        continue
    input = img_treat.load_image_from_path(arg).float().view(1, 3, img_width_down, img_height_down)
    output = cnn.forward(input)
    print("predicted a :", output.item())
    if (output.item() >= 0.5):
        output = 1
    else:
        output = 0
    print("predicted a :", output)