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

def get_training_images_from_directories():
    fake = [PATH_TRAINING_FAKE_CROPPED + f for f in listdir(PATH_TRAINING_FAKE_CROPPED) if ('.jpeg' in f)]
    real = [PATH_TRAINING_REAL_CROPPED + f for f in listdir(PATH_TRAINING_REAL_CROPPED) if ('.jpeg' in f)]
    return (fake, real)

def get_predict_images_from_directories():
    fake = [PATH_PREDICT_FAKE_CROPPED + f for f in listdir(PATH_PREDICT_FAKE_CROPPED) if ('.jpeg' in f)]
    real = [PATH_PREDICT_REAL_CROPPED + f for f in listdir(PATH_PREDICT_REAL_CROPPED) if ('.jpeg' in f)]
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
    all_images = []

    expected = np.zeros((image_size * 2))
    random_index = torch.randperm(image_size * 2)

    for i in range(image_size):
        all_images.append(fake[i])
        expected[i * 2] = 0
        all_images.append(real[i])
        expected[i * 2 + 1] = 1

    return (all_images, torch.tensor(expected).float(), torch.tensor(random_index).view(-1, 64, 1).float())

def reshuffle_data(rand_index):
    rand_index = rand_index.view(-1, 1)
    random_index = torch.randperm(rand_index.size()[0])

    return (torch.tensor(random_index).view(-1, 64, 1).float())

def load_single_batch(img_treat, path, expected, rand_index):
    images_data = np.zeros((64, 3, img_width_down, img_height_down))
    batch_expected = np.zeros((64, 1))

    for i in range(64):
        images_data[i] = img_treat.load_image_from_path(path[int(rand_index[i])])
        batch_expected[i] = expected[int(rand_index[i])]
    return (torch.tensor(images_data).float(), torch.tensor(batch_expected).float())

def create_random_predict_dataset():
    fake, real = get_predict_images_from_directories()
    image_size = len(real)
    all_images = []
    if (len(fake) < len(real)):
        image_size = len(fake)
    if (image_size % 2 != 0):
        image_size -= 1

    expected = np.zeros((image_size * 2))
    random_index = torch.randperm(image_size * 2)

    for i in range(image_size):
        all_images.append(fake[i])
        expected[i * 2] = 0
        all_images.append(real[i])
        expected[i * 2 + 1] = 1

    return (all_images, torch.tensor(expected).float(), torch.tensor(random_index).float())