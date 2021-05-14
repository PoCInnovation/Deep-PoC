from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as dataloader

from env import *
from cnn_model import *
from image_treatment import *

EPOCH = 100

def get_all_images_from_directories(): ## Load les image des dossier
    fake = ["./eye_corpped_fake/" + f for f in listdir("./eye_corpped_fake") if ('.jpeg' in f)]
    real = ["./eye_corpped_real/" + f for f in listdir("./eye_corpped_real") if ('.png' in f)]
    return (fake, real)

def get_max_with_the_current_data(real, fake):
    if (len(fake) - len(fake) % 64 < len(real) - len(real) % 64): ##Je recup le nombre d'image a a load, avec la possibiluter de faire des batch de 64 parfait (sans un batch d'image restante)
        image_size = len(fake) - len(fake) % 64
    else:
        image_size = len(real) - len(real) % 64
    return image_size

def create_random_dataset(): ## Fonction pour regrouper toute les data en Batch
    fake, real = get_all_images_from_directories()

    image_size = get_max_with_the_current_data(real, fake)

    print(image_size)
    
    images_data = np.zeros((image_size * 2, 3, img_width_down, img_height_down)) ##tensor pour stocker toute les "iamge"
    expected = np.zeros((image_size * 2))                                        ##tensor pour stocker si c'est un fake ou pas (1 ou 0)
    random_index = torch.randperm(image_size * 2)                                ##je créer des indice random

    for index_f in range(image_size):
        images_data[random_index[index_f * 2]] = load_image_from_path(fake[index_f])  ##je load une image de deepfake, et je stock cette image dasn une casse random du tensor qui regroupe toute les image
        expected[random_index[index_f * 2]] = 0                                       ##je stock au meme emplacement un 0 pour dire que c'est un fake

        images_data[random_index[index_f * 2 + 1]] = load_image_from_path(real[index_f])##je load une image de real, je stock cette image dasn une casse random du tensor qui regroupe toute les image
        expected[random_index[index_f * 2 + 1]] = 1                                     ##je stock au meme emplacement un 0 pour dire que c'est un vrai

    print(index_f)
    return (torch.tensor(images_data).view(-1, 64, 3, 200, 40), torch.tensor(expected).view(-1, 64, 1).float())##return les valeur en avec un "resize" pour créer des btach de 64

fake, expected = create_random_dataset()

cnn = CNN()


loss_model = nn.BCELoss()
optimizer = torch.optim.Adam(cnn.model.parameters(), lr=0.0001)
for i in range(EPOCH):
    if (i % 5 == 0):
        fake, expected = create_random_dataset()
    for i in range(fake.size()[0]):
        input = fake[i].float()
        input = input.view(1, -1, 3, img_width_down, img_height_down)
        output = cnn.forward(input)
        loss = loss_model(output, expected[i].unsqueeze(0))
        cnn.backward(loss)
        print("loss =", loss)
        optimizer.step()
        optimizer.zero_grad()