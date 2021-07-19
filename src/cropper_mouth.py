from PIL import Image 
import PIL
from os import listdir
from os.path import isfile, join
import sys
from env import *

##fake
mypath = sys.argv[1]
folder_save = sys.argv[2]

left = 292
top = 685
right = left + mouth_img_width
bottom = top + mouth_img_height

def get_all_file():
    file_list = [mypath + f for f in listdir(mypath) if (sys.argv[3] in f)]
    file_list.sort()
    return (file_list)

def crop_image(path):
    print(path)
    im = Image.open(path)
    im = im.crop((left, top, right, bottom))
    path = path.split("/")
    im.save(folder_save + path[-1])

image_list = get_all_file()

print(image_list)
for i in image_list:
    crop_image(str(i))