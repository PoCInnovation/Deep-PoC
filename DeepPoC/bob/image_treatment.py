import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from env import *
from PIL import Image, ImageEnhance
from random import seed
from random import random

class image_treatment():
    def __init__(self):
        self.color, self.contrast, self.brightness, self.sharpness = COLOR, CONTRAST, BRIGHTNESS, SHARPNESS

    def pil_to_tensor(self, image):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        img = img.view(len(image.getbands()), image.size[0], image.size[1])
        return img

    def tensor_rescale_normalize(self, tens):
        trans = transforms.Compose([transforms.Resize((img_width_down, img_height_down))])
        return trans(tens)

    def create_contraste(self, image):
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(self.color)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.contrast)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.brightness)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self.sharpness)
        return image

    def load_image_from_path(self, path_name):
        im = Image.open(path_name).convert('RGB')
        im = self.create_contraste(im)
        im = self.pil_to_tensor(im)
        img = self.tensor_rescale_normalize(im)
        return img