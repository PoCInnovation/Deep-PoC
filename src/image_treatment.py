from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as dataloader
import time
from env import *
from PIL import Image, ImageEnhance
from random import seed
from random import random

def pil_to_tensor(image): ## Transforme une image en tensor
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    img = img.view(len(image.getbands()), image.size[0], image.size[1])
    return img

def tensor_rescale_normalize(tens):  ## Normalize la taille de l'image en un 200 * 40
    trans = transforms.Compose([transforms.Resize((img_width_down,img_height_down))])
    return trans(tens)

def create_contraste(image, color = 1, contrast = 1, brightness = 1, sharpness = 1):## Fonction pour créer les contraste / modifier l'image 
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    # image.save("thumbnail.jpeg")
    return image

def load_image_from_path(path_name):
    im = Image.open(path_name)
    im = create_contraste(im, 2, 1, 1, 1)
    im = pil_to_tensor(im)
    img = tensor_rescale_normalize(im)
    return img