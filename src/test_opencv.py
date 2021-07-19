from time import sleep
import cv2 as cv
import numpy as np
import yolo_test_utils

IMG_WIDTH = 416
IMG_HEIGHT = 416

cap = cv.VideoCapture(0)

net = cv.dnn.readNetFromDarknet("yolo_params/yolov3-face.cfg", "yolo_params/yolov3-wider_16000.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

while True:
    ret, frame = cap.read()
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layers_names = net.getLayerNames()
    ourputs_layer = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(ourputs_layer)

    faces = yolo_test_utils.post_process(frame, outs, yolo_test_utils.CONF_THRESHOLD, yolo_test_utils.NMS_THRESHOLD)

    info = [
        ('number of faces detected', '{}'.format(len(faces)))
    ]

    for (i, (txt, val)) in enumerate(info):
        text = '{}: {}'.format(txt, val)
        cv.putText(frame, text, (10, (i * 20) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, yolo_test_utils.COLOR_RED, 2)

    # Save the output video to file
    # if args.image:
    #     cv.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
    # else:
    #     video_writer.write(frame.astype(np.uint8))

    cv.imshow('frame', frame)
    cv.waitKey(1)