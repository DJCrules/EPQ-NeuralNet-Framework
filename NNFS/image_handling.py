from PIL import Image
import glob, os
from random import randint
import numpy as np
import pandas as pd

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

def randomise(arr):
    for i in range(len(arr) - 1, 0, -1):
        j = randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
        if j > len(arr):
            print("randomised")
    return arr


def get_meta_data(set_num):
    index_range = find_index_range(set_num)
    a = 0
    data = []
    df = pd.read_csv(r"C:\Users\dylan\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3\Data_Entry_2017.csv")
    for index, row in df.iterrows():
        if str(row['Image Index']) == index_range[0]:
            a = 1
        if str(row['Image Index']) == index_range[1]:
            return data
        if a == 1:
            data.append(row["Finding Labels"])
    return data

def convert_meta_data(data):
    #Organisation of metadata:
    #[Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Nodule,
    #  Mass, Pneumothorax, Pleural_Thickening, Atelectasis, Fibrosis, 
    #  Consolidation, Edema]
    converted_data = []
    i = 0
    for part in data:
        converted_data.append([])
        if 'Cardiomegaly' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Emphysema' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Effusion' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Hernia' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Infiltration' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Nodule' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Mass' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Pneumothorax' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Pleural_Thickening' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Atelectasis' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Fibrosis' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Consolidation' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Edema' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Atelectasis' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'No Finding' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        i+=1
    return converted_data
        