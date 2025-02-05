from PIL import Image
import glob, os
from core import randomise

def image_conversion(fileName):
    with Image.open(fileName) as im:
        image = im.getdata()
    return image

def images_conversion(fileName):
    images = []
    for image in glob.glob(fileName):
        images.append(image_conversion(image))
    return images

def find_images(set_num, specific = "*"):
    # Download latest version
    # path = kagglehub.dataset_download("nih-chest-xrays/data")
    if set_num < 10:
        return r"C:\Users\dylan\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3" + r"\images_00" + str(set_num) + r"\images\\" + specific
    elif 100 > set_num > 9:
        return r"C:\Users\dylan\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3" + r"\images_0" + str(set_num) + r"\images\\" + specific

def fetch_image_set(n, specific = "*"):
    print("fetching address")
    images = find_images(n, specific)
    print("fetching images from " + str(images))
    converted_images = images_conversion(images)
    print("randomising images")
    randomised_images = randomise(converted_images)
    print("done")
    return randomised_images

def find_index_range(set_num):
    setvar = find_images(set_num)
    images = glob.glob(setvar)
    return os.path.basename(images[0]), os.path.basename(images[len(images) - 1])