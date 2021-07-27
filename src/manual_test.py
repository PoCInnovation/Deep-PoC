import sys
from os import path
from env import *
from cnn_model import *
from image_treatment import *

def get_metadata_from_weights(weights_path):
    weights_path = weights_path.split('/')[-1]
    weights_path = weights_path.split('_')[-1]
    metadata = weights_path.split(',')
    return [float(metadata[0]), float(metadata[1]), float(metadata[2]), float(metadata[3])]

def usage():
    print("\nThis script lets tou test if an image is a deepfake (0 = deepfake, 1 = real Image)\n")
    print("Usage:\tmanual_test.py [eye_weight] [mouth_weight] [eye_image] [mouth_image]\n")
    print("\t-eye_weight:      The pre-trained eye weights")
    print("\t-mouth_weight:    The pre-trained mouth weights")
    print("\t-eye_image:       The eyes of the image")
    print("\t-mouth_image:     The mouth of the images\n")

def error():
    if (len(sys.argv) == 1):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        exit(1)
    if (len(sys.argv) != 5):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (not path.isfile(sys.argv[3]) or not path.isfile(sys.argv[4])):
        print("Not a real path, run with '-h' for help")
        exit(84)

def print_output(output, face_part):
    print(face_part, " result: ", output.item())


def main():
    error()
    eye_weight = sys.argv[1]
    mouth_weight = sys.argv[2]
    cnn = CNN()
    img_treat = image_treatment()
    eye_metadata = get_metadata_from_weights(eye_weight)
    mouth_metadata = get_metadata_from_weights(mouth_weight)
    print(eye_metadata)
    print(mouth_metadata)

#EYE
    cnn.load_state_dict(torch.load(eye_weight))
    img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness = eye_metadata[0], eye_metadata[1], eye_metadata[2], eye_metadata[3]
    input = img_treat.load_image_from_path(sys.argv[3]).float().view(1, 3, img_width_down, img_height_down)
    output = cnn.forward(input)
    print_output(output, "Eye")
#MOUTH
    cnn.load_state_dict(torch.load(mouth_weight))
    img_treat.color, img_treat.contrast, img_treat.brightness, img_treat.sharpness = mouth_metadata[0], mouth_metadata[1], mouth_metadata[2], mouth_metadata[3]
    input = img_treat.load_image_from_path(sys.argv[4]).float().view(1, 3, img_width_down, img_height_down)
    output = cnn.forward(input)
    print_output(output, "Mouth")



if __name__ == "__main__":
    main()