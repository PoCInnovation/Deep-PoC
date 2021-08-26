from urllib import request
from os import listdir, path
from time import sleep
import sys

src = 'https://thispersondoesnotexist.com/image'

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def get_all_file_nbr(path):
    file_list = [int(f[:-5]) for f in listdir(path) if ('.jpeg' in f and is_integer(f[:-5]))]
    file_list.sort()
    return (file_list)

def create_new_image(path):
    sleep(1)
    file_id_list = get_all_file_nbr(path)
    for i, elmnt in enumerate(file_id_list):
        if (i != elmnt):
            request.urlretrieve(src, path + str(i) + '.jpeg')
            print("created: ", str(i) + '.jpeg')
            return
    request.urlretrieve(src, path + str(len(file_id_list)) + '.jpeg')
    print("created: ", str(len(file_id_list)) + '.jpeg')

def usage():
    print("\nThis script lets you extract image from https://thispersondoesnotexist.com. to a given path\n")
    print("Usage:\tget_fake_pic.py [Path] [Iteration]\n")
    print("\t-Path:         Where to save the images (String)")
    print("\t-Iteration:    The number of images you want to extract (Int)\n")

def error():
    if (len(sys.argv) == 1):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        exit(1)
    if (len(sys.argv) != 3):
        print("Wrong number of arguments, run with '-h' or '--help' for help")
        exit(84)
    if not (path.isdir(sys.argv[1])):
        print("Not a real path, run with '-h' for help")
        exit(84)
    if not (is_integer(sys.argv[2])):
        print("Not a correct number, run with '-h' for help")
        exit(84)

def main():
    error()
    path = sys.argv[1]
    if (path[-1] != "/"):
        path = path + "/"
    iterations = int(sys.argv[2])
    opener = request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    request.install_opener(opener)
    for i in range(iterations):
        create_new_image(path)


if __name__ == "__main__":
    main()