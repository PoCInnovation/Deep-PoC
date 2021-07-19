from urllib import request
from selenium import webdriver
from os import listdir
from os.path import isfile, join
from time import *
import sys

mypath = sys.argv[1]
src = 'https://thispersondoesnotexist.com/image'


opener = request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
request.install_opener(opener)


def get_all_file_nbr():
    file_list = [int(f[:-5]) for f in listdir(mypath) if ('.jpeg' in f)]
    file_list.sort()
    return (file_list)

def create_new_image():
    sleep(1)
    file_id_list = get_all_file_nbr()
    for i, elmnt in enumerate(file_id_list):
        if (i != elmnt):
            request.urlretrieve(src, mypath + str(i) + '.jpeg')
            print("created: ", str(i))
            return
    request.urlretrieve(src, mypath + str(len(file_id_list)) + '.jpeg')
    print("created: ", str(len(file_id_list)))

for i in range(1000):
    create_new_image()