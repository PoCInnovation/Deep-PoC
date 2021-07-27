from PIL import Image
import sys
from os import path, getcwd
sys.path.insert(0, getcwd() + "/src")

from image_treatment import *

def is_float(n):
    try:
        float(n)
    except ValueError:
        return False

def usage():
    print("\nThis script lets you test different metadata applied to given images\n")
    print("Usage:\ttest_metadata.py [Color] [Contrast] [Brightness] [Sharpness] [Image_0] [Image_1] ... [Image_n]\n")
    print("\t-Color:        (Float)(see: https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)")
    print("\t-Contrast:     (Float)(see: https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)")
    print("\t-Brightness:   (Float)(see: https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)")
    print("\t-Sharpness:    (Float)(see: https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)")
    print("\t-Image_n:      (String)The path to an image to test\n")

def error():
    if (len(sys.argv) == 1):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        exit(1)
    if (len(sys.argv) > 6):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (is_float(sys.argv[1]) or is_float(sys.argv[2]) or is_float(sys.argv[3]) or is_float(sys.argv[4])):
        print("Wrong type of auguments, run with '-h' or '--help' for help")
        exit(84)

def main():
    error()
    color = float(sys.argv[1])
    contrast = float(sys.argv[2])
    brightness = float(sys.argv[3])
    Sharpness = float(sys.argv[4])
    for i, elmnt in enumerate(sys.argv):
        if (i < 5):
            continue
        if not (path.isfile(sys.argv[i])):
            print("{} is not a correct path, skipped".format(sys.argv[i]))
        image = Image.open(elmnt)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(Sharpness)
        image.save("test_contrast_" + elmnt.split("/")[-1].split(".")[0] + ".jpeg")
        print("Created: {}".format("test_contrast_" + elmnt.split("/")[-1].split(".")[0] + ".jpeg"))


if __name__ == "__main__":
    main()