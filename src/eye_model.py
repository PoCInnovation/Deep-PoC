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

EPOCH = 10
seed(1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), padding=(0, 2, 2)),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 4, 4), padding=(0, 1, 1)),
            nn.MaxPool3d((1, 2, 2)),
            nn.Flatten(2),
            nn.Linear(1500, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def backward(self, loss):
        loss.backward()


def get_all_images_from_directories(): ## Load les image des dossier
    fake = ["./eye_corpped_fake/" + f for f in listdir("./eye_corpped_fake") if ('.jpeg' in f)]
    real = ["./eye_corpped_real/" + f for f in listdir("./eye_corpped_real") if ('.png' in f)]
    return (fake, real)

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
    return image

def load_image_from_path(path_name):
    im = Image.open(path_name)
    im = create_contraste(im)
    im = pil_to_tensor(im)
    img = tensor_rescale_normalize(im)
    return img

def create_random_dataset(): ## Fonction pour regrouper toute les data en Batch //TRUC QUI RISQUE ETRE NUL
    fake, real = get_all_images_from_directories()
    image_size = 0
    if (len(fake) - len(fake) % 64 < len(real) - len(real) % 64): ##Je recup le nombre d'image a a load, avec la possibiluter de faire des batch de 64 parfait (sans un batch d'image restante)
        image_size = len(fake) - len(fake) % 64
    else:
        image_size = len(real) - len(real) % 64

    print(image_size)
    
    images_data = np.zeros((image_size * 2, 3, img_width_down, img_height_down)) ##tensor pour stocker toute les "iamge"
    expected = np.zeros((image_size * 2)) ##tensor pour stocker si c'est un fake ou pas (1 ou 0)
    random_index = torch.randperm(image_size * 2) ##je créer des indice random

    for index_f in range(image_size):
        images_data[random_index[index_f * 2]] = load_image_from_path(fake[index_f])  ##je load une image de deepfake, et je stock cette image dasn une casse random du tensor qui regroupe toute les image
        expected[random_index[index_f * 2]] = 0 ##je stock au meme emplacement un 0 pour dire que c'est un fake

        im = Image.open(real[index_f])
        images_data[random_index[index_f * 2 + 1]] = load_image_from_path(real[index_f])##je load une image de real, je stock cette image dasn une casse random du tensor qui regroupe toute les image
        expected[random_index[index_f * 2 + 1]] = 1 ##je stock au meme emplacement un 0 pour dire que c'est un vrai

    print(index_f)
    print(torch.tensor(images_data).size())
    return (torch.tensor(images_data).view(-1, 64, 3, 200, 40), torch.tensor(expected).view(-1, 64, 1).float())##return les valeur en avec un "resize" pour créer des btach de 64

fake, expected = create_random_dataset()
print(fake.size())
print(expected.size())

cnn = CNN()


loss_model = nn.BCELoss()
optimizer = torch.optim.Adam(cnn.model.parameters(), lr=0.00001)
for i in range(EPOCH):   ##la je train l'ia
    for i in range(fake.size()[0]):
        input = fake[i].float()
        input = input.view(1, -1, 3, img_width_down, img_height_down)
        output = cnn.forward(input)
        print(output.size())
        print("first val:", {output[0][0]})
        loss = loss_model(output, expected[i].unsqueeze(0))
        cnn.backward(loss)
        print(loss)
        optimizer.step()
        optimizer.zero_grad()