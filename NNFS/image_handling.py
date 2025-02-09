from PIL import Image
import glob, os
from random import randint
import numpy as np

def image_file_conversion(fileName):
    #Take the images from secondary to primary storage
    with Image.open(fileName) as im:
        image = list(im.getdata())
        image_array = np.array(image, dtype=np.float32)
        image_array /= 255
    return image_array

def image_conversion(image_):
    image = list(image_)
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0
    return image_array

def images_conversion(fileName):
    #Take multiple images from secondary to primary storage
    images = []
    for image in glob.glob(fileName):
        images.append(image_file_conversion(image))
    return images

def find_images(set_num, specific = "*"):
    # Download latest version
    # path = kagglehub.dataset_download("nih-chest-xrays/data")
    if set_num < 10:
        return r"C:\Users\dylan\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3" + r"\images_00" + str(set_num) + r"\images\\" + specific
    elif 100 > set_num > 9:
        return r"C:\Users\dylan\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3" + r"\images_0" + str(set_num) + r"\images\\" + specific

def fetch_image_set(n, specific = "*"):
    #get a set of images from the n set
    print("fetching address")
    images = find_images(n, specific)
    print("fetching images from " + str(images))
    converted_images = images_conversion(images)
    print("randomising images")
    randomised_images = randomise(converted_images)
    print("done")
    return randomised_images

def find_index_range(set_num):
    #find the numbers associated with a bunch of image files
    setvar = find_images(set_num)
    images = glob.glob(setvar)
    return os.path.basename(images[0]), os.path.basename(images[len(images) - 1])

def find_metadata(set_num, specific = "*"):
    return set_num

def randomise(arr):
    for i in range(len(arr) - 1, 0, -1):
        j = randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
        if j > len(arr):
            print("randomised")
    return arr