from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import time
from env import *
from PIL import Image, ImageEnhance
from random import seed
from random import random

seed(1)

# class simple_mod(nn.Module):
#     def __init__(self):
#         super(simple_mod, self).__init__()
#         self.loss_model = nn.Sigmoid()
#         self.model = nn.Sequential(
#             nn.Linear(24000, 2400),
#             nn.ReLU(),
#             nn.Linear(2400, 1200),
#             nn.ReLU(),
#             nn.Linear(1200, 600),
#             nn.ReLU(),
#             nn.Linear(600, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         x = x.float()
#         x = x.view(24000)
#         x = self.model.forward(x)
#         return x

#     def backward(self, loss):
#         loss.backward()

#     def predict(self, data):
#         outputs = self.forward(data)
#         return outputs

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.loss_model = nn.Sigmoid()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(1, 3, 3), padding=(0, 2, 2)),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(in_channels=2, out_channels=4, kernel_size=(1, 4, 4), padding=(0, 1, 1)),
            nn.MaxPool3d((1, 2, 2)),
            nn.Flatten(),
            nn.Linear(6000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()
        x = x.view(1, 1, 3, img_width_down, img_height_down)
        x = self.model.forward(x)
        return x

    def backward(self, loss):
        loss.backward()


def load_image():
    fake = ["./eye_corpped_fake/" + f for f in listdir("./eye_corpped_fake") if ('.jpeg' in f)]
    real = ["./eye_corpped_real/" + f for f in listdir("./eye_corpped_real") if ('.png' in f)]
    return (fake, real)

def pil_to_tensor(image):
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    img = img.view(len(image.getbands()), image.size[0], image.size[1])
    return img

def tensor_rescale(tens):
    trans = transforms.Compose([transforms.Resize((img_width_down,img_height_down))])
    return trans(tens)

def create_contraste(image, color = 1, contrast = 1, brightness = 1, sharpness = 1):
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    return image

def get_data():
    fake, real = load_image()
    fake_list = np.zeros((len(fake), 3, img_width_down, img_height_down))
    for j, i in enumerate(fake):
        im = Image.open(i)
        im = create_contraste(im, 2, 2)
        im = pil_to_tensor(im)
        img = tensor_rescale(im)
        fake_list[j] = img
    print(torch.tensor(fake_list).size())
    real_list = np.zeros((len(real), 3, img_width_down, img_height_down))
    for j, i in enumerate(real):
        im = Image.open(i)
        im = create_contraste(im, 2, 2)
        im = pil_to_tensor(im)
        img = tensor_rescale(im)
        real_list[j] = img
    print(torch.tensor(real_list).size())
    return (torch.tensor(fake_list), torch.tensor(real_list))

fake, real = get_data()

cnn = CNN()


loss_model = nn.MSELoss()
optimizer = torch.optim.SGD(cnn.model.parameters(), lr=0.01)
for i in range(500):
    i = random() % 2
    if (i == 0):
        output = cnn.forward(fake[int(random() % len(fake))])
    else:
        output = cnn.forward(real[int(random() % len(real))])
    print(output.size())
    print(torch.tensor(i).unsqueeze(0))
    loss = loss_model(output, torch.tensor(i).unsqueeze(0).unsqueeze(0))
    cnn.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    break


##bash, adam, beautiffy, visual representation, notion