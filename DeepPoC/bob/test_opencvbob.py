from time import sleep
import cv2 as cv
import numpy as np
import yolo_test_utils
import sys
import torch

from env import *
from cnn_model import *
from image_treatment import *
from image_gest import *
from benchmark_excel import *

IMG_WIDTH = 416
IMG_HEIGHT = 416

def main_opencv(weights_path):
    img_treat = image_treatment()
    cnn = CNN()
    cnn.load_state_dict(torch.load("../eye_model/10.0,1.0,1.0,10.0"))
    weights = weights_path
    weights = weights.split('/')[-1]
    weights = weights.split(',')
    img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness = float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3])
    cap = cv.VideoCapture("vieo.mp4")

    net = cv.dnn.readNetFromDarknet("../yolo_params/yolov3-face.cfg", "../yolo_params/yolov3-wider_16000.weights")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    addition = 0
    total = 0
    while total < 10:
        ret, frame = cap.read()
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layers_names = net.getLayerNames()
        ourputs_layer = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(ourputs_layer)

        faces = yolo_test_utils.post_process(frame, outs, yolo_test_utils.CONF_THRESHOLD, yolo_test_utils.NMS_THRESHOLD)
        if (len(faces) != 0):
            addition += yolo_test_utils.prediction_from_ai(img_treat, cnn)
            total += 1

        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv.putText(frame, text, (10, (i * 20) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, yolo_test_utils.COLOR_RED, 2)

        # cv.imshow('frame', frame)
        cv.waitKey(1)
    if (total != 0):
        return addition / total

if __name__ == "__main__":
    main_opencv()