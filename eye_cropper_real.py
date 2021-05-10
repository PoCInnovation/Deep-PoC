from PIL import Image 
import PIL
from os import listdir
from os.path import isfile, join

##eal
mypath = './face_real/'
folder_save = './eye_corpped_real/'

left = 287
top = 440
right = left + 450
bottom = top + 100

def get_all_file():
    file_list = [mypath + f for f in listdir(mypath) if ('.png' in f)]
    file_list.sort()
    return (file_list)

def crop_image(path):
    im = Image.open(path)
    im = im.crop((left, top, right, bottom))
    path = path.split("/")
    im.save(folder_save + path[-1])

image_list = get_all_file()

print(image_list)
for i in image_list:
    crop_image(str(i))