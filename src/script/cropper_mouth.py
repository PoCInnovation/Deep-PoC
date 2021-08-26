from PIL import Image 
from os import listdir, path, getcwd
import sys

sys.path.insert(0, getcwd() + "/src")

from env import *

left = 292
top = 685
right = left + mouth_img_width
bottom = top + mouth_img_height

def get_all_file(original_path, image_type):
    file_list = [original_path + f for f in listdir(original_path) if (image_type in f)]
    file_list.sort()
    return file_list

def crop_image(path, destination_path):
    print(path)
    im = Image.open(path)
    im = im.crop((left, top, right, bottom))
    path = path.split("/")
    im.save(destination_path + "mouth_" + path[-1])
    print("Cropped: {}".format(destination_path + "mouth_" + path[-1]))

def usage():
    print("\nThis script lets you crop the mouth of all images in a path.\n")
    print("Usage:\tcropper_mouth.py [original_path] [destination_path] [Image_type]\n")
    print("\t-original_path:       The path from where to fetch the images(String)")
    print("\t-destination_path:    The path to store the images(String)")
    print("\t-Image_type:          The type of the images\n")

def error():
    if (len(sys.argv) == 1):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        exit(1)
    if (len(sys.argv) != 4):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (not path.isdir(sys.argv[1]) or not path.isdir(sys.argv[2])):
        print("Not a real path, run with '-h' for help")
        exit(84)

def main():
    error()
    original_path = sys.argv[1]
    destination_path = sys.argv[2]
    image_type = sys.argv[3]

    if (original_path[-1] != "/"):
        original_path = original_path + "/"
    if (destination_path[-1] != "/"):
        destination_path = destination_path + "/"
    if (image_type[0] != "."):
        image_type = "." + image_type

    image_list = get_all_file(original_path, image_type)
    print(image_list)
    for i in image_list:
        crop_image(str(i), destination_path)


if __name__ == "__main__":
    main()