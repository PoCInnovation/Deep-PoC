from time import sleep
import datetime
import numpy as np
import cv2
import sys
import torch
from os import listdir
import os

from env import *
from cnn_model import *
from image_treatment import *
from image_gest import *
from benchmark_excel import *

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


def get_outputs_names(net):
    layers_names = net.getLayerNames()

    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def prediction_from_ai(img_treat, cnn):
    if not os.path.isfile("./eye_cv.jpeg"):
        return
    input = img_treat.load_image_from_path("./eye_cv.jpeg").float().view(1, 3, img_width_down, img_height_down)
    output = cnn.forward(input)
    print("predicted a :", output.item())
    if (output.item() >= 0.5):
        output = 1
    else:
        output = 0
    print("predicted a :", output)

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def save_image(prefix, img):
    file_list = [int(f[:-5]) for f in listdir(".") if ('.jpeg' in f) and is_integer(f[:-5])]
    file_list.sort()
    cv2.imwrite(prefix + str(len(file_list)) + ".jpeg", img)

def draw_predict(frame, conf, left, top, right, bottom):
    sizey = bottom - top
    sizex = right - left
    crop_img = frame[top:bottom, left:right]
    if (crop_img.size != 0):
        cv2.imwrite('face_cv.jpeg', crop_img)
        # save_image("1100", crop_img)
        eyes = crop_img[int(top*0.431):int(top*0.431+sizey*0.135), left*0:left*0+sizex*1]
        if (eyes.any()):
            cv2.imwrite('eye_cv.jpeg', eyes)
            # save_image("2100", eyes)
        mouth = crop_img[int(top*0.920):int(top*0.920+sizey*0.193), left*0:left*0+sizex*1]
        if (mouth.any()):
            cv2.imwrite('mouth_cv.jpeg', mouth)
            # save_image("3100", mouth)

    text = '{:.2f}'.format(conf)

    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        draw_predict(frame, confidences[i], left, top, right, bottom)
    return final_boxes

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height
    return left, top, right, bottom
