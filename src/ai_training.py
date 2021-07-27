import torch
import torch.nn as nn
import csv
import sys

from env import *
from cnn_model import *
from image_treatment import *
from image_gest import *
from benchmark_excel import *

cnn_first = CNN()
excel_write = excel_writer()
img_treat = image_treatment()

loss_model = nn.BCELoss()

def training(cnn, loss_model, optimizer):
    path, expected, rand_index = create_random_training_dataset(sys.argv[1])
    excel_write.add_new_benchmark(img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness)
    for j in range(EPOCH):
        if (j % 5 == 0):
            rand_index = reshuffle_data(rand_index)
        loss_av = 0
        total = 0
        for i in range(rand_index.size()[0]):
            batch_image, batch_expected = load_single_batch(img_treat, path, expected, rand_index[i])
            input = batch_image.float()
            output = cnn.forward(input)
            loss = loss_model(output, batch_expected)
            loss_av += loss
            total += 1
            cnn.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        if (total != 0):
            print("loss after ", j, "EPOCH : ", loss_av / total)
            excel_write.add_new_line((loss_av / total).item())
        excel_write.workbook.save('benchmark.xlsx')
    torch.save(cnn.state_dict(), "{}_{},{},{},{}".format(sys.argv[1], img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness))

def predict(cnn):
    overall = 0
    data, expected, rand_index = create_random_predict_dataset(sys.argv[1])
    for i in rand_index:
        input = img_treat.load_image_from_path(data[int(i)]).float().view(1, 3, img_width_down, img_height_down)
        output = cnn.forward(input)
        if (output.item() >= 0.5 and expected[int(i)].item() >= 0.5):
            overall += 1
        if (output.item() < 0.5 and expected[int(i)].item() < 0.5):
            overall += 1
        print("predicted : ", output.item(), "expected : ", expected[int(i)].item())
    print(overall, "out of", rand_index.size()[0])
    excel_write.add_new_line("{} out of {}".format(overall, rand_index.size()[0]))
    excel_write.add_new_line("{}%".format(overall * 100 / rand_index.size()[0]))

def benchmark():
    file = open("image_trans_tests.csv", "r")
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if (i == 0):
            continue
        cnn_first = CNN()
        optimizer_first = torch.optim.Adam(cnn_first.model.parameters(), lr=0.0002)
        img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        print(img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness)
        training(cnn_first, loss_model, optimizer_first)
        predict(cnn_first)

def usage():
    print("\nThis script lets you run the training for the AI, with the input taking from image_trans_tests.csv.\n")
    print("Usage:\tai_training.py [facial_part]\n")
    print("\t-facial_part:    The part of the face on which to train the AI ('eyes' or 'mouth')\n")

def error():
    if (len(sys.argv) == 1):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        exit(1)
    if (len(sys.argv) != 2):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (sys.argv[1] != 'eyes' and sys.argv[1] != 'mouth'):
        print("Wrong argument, must be 'eyes' or 'mouth', run with '-h' or '--help' for help")
        exit(84)

def main():
    error()
    benchmark()
    excel_write.workbook.save('benchmark.xlsx')


if __name__ == "__main__":
    main()