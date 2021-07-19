from time import sleep
import datetime
import numpy as np
import cv2

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

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


def draw_predict(frame, conf, left, top, right, bottom):
    sizey = bottom - top
    sizex = right - left
    crop_img = frame[top:bottom, left:right]
    if (crop_img.size != 0):
        print("bob")
        cv2.imwrite('/home/slooth/pytorchtest/face_cv.jpg', crop_img)
        eyes = crop_img[int(top*0.431):int(top*0.431+sizey*0.135), left*0:left*0+sizex*1]
        cv2.imwrite('/home/slooth/pytorchtest/eye_cv.jpg', eyes)
        mouth = crop_img[int(top*0.920):int(top*0.920+sizey*0.193), left*0:left*0+sizex*1]
        cv2.imwrite('/home/slooth/pytorchtest/mouth_cv.jpg', mouth)

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
