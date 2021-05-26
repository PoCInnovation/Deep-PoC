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

EPOCH = 20

def get_training_images_from_directories():
    fake = [PATH_TRAINING_FAKE_CROPPED + f for f in listdir(PATH_TRAINING_FAKE_CROPPED) if ('.jpeg' in f)]
    real = [PATH_TRAINING_REAL_CROPPED + f for f in listdir(PATH_TRAINING_REAL_CROPPED) if ('.png' in f)]
    return (fake, real)

def get_predict_images_from_directories():
    fake = [PATH_PREDICT_FAKE_CROPPED + f for f in listdir(PATH_PREDICT_FAKE_CROPPED) if ('.jpeg' in f)]
    real = [PATH_PREDICT_REAL_CROPPED + f for f in listdir(PATH_PREDICT_REAL_CROPPED) if ('.png' in f)]
    return (fake, real)


def get_max_with_the_current_data(real, fake):
    image_size = 0
    if (len(fake) - len(fake) % 64 < len(real) - len(real) % 64):
        image_size = len(fake) - len(fake) % 64
    else:
        image_size = len(real) - len(real) % 64
    return (image_size)

def create_random_training_dataset():
    fake, real = get_training_images_from_directories()
    image_size = get_max_with_the_current_data(real, fake)
    print(image_size)

    images_data = np.zeros((image_size * 2, 3, img_width_down, img_height_down))
    expected = np.zeros((image_size * 2))
    random_index = torch.randperm(image_size * 2)

    for index_f in range(image_size):
        images_data[random_index[index_f * 2]] = load_image_from_path(fake[index_f])
        expected[random_index[index_f * 2]] = 0

        images_data[random_index[index_f * 2 + 1]] = load_image_from_path(real[index_f])
        expected[random_index[index_f * 2 + 1]] = 1

    print(index_f)
    return (torch.tensor(images_data).view(-1, 64, 3, 200, 40), torch.tensor(expected).view(-1, 64, 1).float())

def reshuffle_data(fake, expected):
    fake = fake.view(-1, 3, 200, 40)
    expected = expected.view(-1, 1)
    image_size = fake.size()[0]
    print(image_size)
    images_data = np.zeros((image_size, 3, img_width_down, img_height_down))
    images_expected = np.zeros((image_size))
    random_index = torch.randperm(image_size)

    for index_f in range(image_size):
        images_data[random_index[index_f]] = fake[index_f]
        images_expected[random_index[index_f]] = expected[index_f]
    return (torch.tensor(images_data).view(-1, 64, 3, 200, 40), torch.tensor(images_expected).view(-1, 64, 1).float())


def create_random_predict_dataset():
    fake, real = get_predict_images_from_directories()
    image_size = len(real)
    if (len(fake) < len(real)):
        image_size = len(fake)
    if (image_size % 2 != 0):
        image_size -= 1
    print(image_size)
    
    images_data = np.zeros((image_size * 2, 3, img_width_down, img_height_down))
    expected = np.zeros((image_size * 2))
    random_index = torch.randperm(image_size * 2)

    for index_f in range(int(image_size)):
        images_data[random_index[index_f * 2]] = load_image_from_path(fake[index_f])
        expected[random_index[index_f * 2]] = 0

        images_data[random_index[index_f * 2 + 1]] = load_image_from_path(real[index_f])
        expected[random_index[index_f * 2 + 1]] = 1

    print(index_f)
    return (torch.tensor(images_data).view(-1, 1, 3, 200, 40), torch.tensor(expected).view(-1, 1, 1).float())



cnn = CNN()

loss_model = nn.BCELoss()
optimizer = torch.optim.Adam(cnn.model.parameters(), lr=0.0002)

def training(cnn, loss_model, optimizer):
    fake, expected = create_random_training_dataset()
    for i in range(EPOCH):
        if (i % 5 == 0):
            fake, expected = reshuffle_data(fake, expected)
        for i in range(fake.size()[0]):
            input = fake[i].float()
            output = cnn.forward(input)
            loss = loss_model(output, expected[i])
            cnn.backward(loss)
            print("loss =", loss)
            optimizer.step()
            optimizer.zero_grad()

def predict(cnn):
    overall = 0
    fake, expected = create_random_predict_dataset()
    print(fake.size(), expected.size())
    for i in range(fake.size()[0]):
        input = fake[i].float()
        output = cnn.forward(input)
        if (output.item() >= 0.5 and expected[i].item() >= 0.5):
            overall += 1
        if (output.item() < 0.5 and expected[i].item() < 0.5):
            overall += 1
        print(output.item(), expected[i].item(), overall, "out of", i)

training(cnn, loss_model, optimizer)
predict(cnn)