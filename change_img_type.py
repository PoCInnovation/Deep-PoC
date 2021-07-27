from os import listdir
from PIL import Image
import sys

def usage():
    print("\nThis script lets you transform a given image type to another image type\n")
    print("Usage:\tchange_img_type.py [Path] [Original_type] [Requested_type]\n")
    print("\t-Path:              Where the images are saved (String)")
    print("\t-Original_type:     The current type of images(String)")
    print("\t-Requested_type:    The requested type of images(String)\n")

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

def main():
    error()
    path = sys.argv[1]
    original_type = sys.argv[2]
    requested_type = sys.argv[3]

    if (path[-1] != "/"):
        path = path + "/"
    if (original_type[0] != "."):
        original_type = "." + original_type
    if (requested_type[0] != "."):
        requested_type = "." + requested_type

    file_list = [path + f for f in listdir(path) if (original_type in f)]
    for i in range(len(file_list)):
        image = Image.open(file_list[i])
        image.save(file_list[i][:-len(original_type)] + requested_type)
        print("Transformed: {} --> {}".format(file_list[i], file_list[i][:-len(original_type)] + requested_type))


if __name__ == "__main__":
    main()