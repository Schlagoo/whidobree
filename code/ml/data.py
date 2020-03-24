""" author:     Pascal Schlaak
    content:    Class to handle data
    python:     3.8.1
"""

import os
import cv2
import shutil
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# PATH_TO_DATA = "/mnt/e/data/stanford_dog_breed/images"
PATH_TO_DATA = "/mnt/e/data/bing_dog_breeds"
PATH_TO_EXAMPLES = "/mnt/e/data/stanford_dog_breed/example_images"
PATH_TO_PREPROCESSED_DATA = "/mnt/e/data/stanford_dog_breed/images_preprocessed_50x50_grayscale_categorical/"
IMG_SIZE = 150


class Data():

    def __init__(self):
        pass
    
    @staticmethod
    def assign_classes():

        """ Write txt-file with breed names and their ids
        """

        class_counter = 1
    	
        files = os.listdir(PATH_TO_DATA)
        # Create new file with id, breed name
        with open("classes.csv", "w+") as f:
            for file in files:
                f.write(str(class_counter) + ", " + file[:9] + ", " + file[10:].lower() + "\n")
                class_counter += 1

        print("File written")

    @staticmethod
    def fetch_breed_example_picture():

        """ Fetch an example picture of every dog breed
        """

        folder_classes = os.listdir(PATH_TO_DATA)
        
        for directory in folder_classes:
            folder_images = os.listdir(os.path.join(PATH_TO_DATA, directory))
            for image in folder_images:
                shutil.copy(os.path.join(PATH_TO_DATA, directory, image), os.path.join(PATH_TO_EXAMPLES, directory[10:].lower() + ".jpg"))
                continue


    def write_pickle(self):

        """ Load images from files
        """

        dataset = []
        folder_classes = os.listdir(PATH_TO_DATA)

        # Append [class, image] of dataset as entry of new array
        for directory in folder_classes:
            folder_images = os.listdir(os.path.join(PATH_TO_DATA, directory))
            for image in folder_images:
                data = self.preprocess(image, directory)
                idx = directory[:2]
                idx = idx.strip("_")
                dataset.append([idx, data])

        print(len(dataset))
        
        # Write dataset to pickle
        with open("/mnt/e/data/bing_dog_breeds_rgb_150x150.pickle", "wb") as pickle_dataset:
            pickle.dump(dataset, pickle_dataset)

        print("File written")

    def replace_class_label(self, label: str) -> int:
        
        """ Replace original class label with idx (int)
        """

        with open("classes.csv", "r") as f:
            for entry in f:
                idx, name, _ = entry.split(", ")
                if name == label:
                    return idx 

    def rename_directories(self):

        """ Rename directories of class images to suit categorical ids
        """

        folder_classes = os.listdir(PATH_TO_DATA)

        # Append [class, image] of dataset as entry of new array
        for directory in folder_classes:
            new_directory = directory[:2]
            new_directory = new_directory.strip("_")
            new_directory = int(new_directory) - 1
            new_directory = str(new_directory) + "_" + directory[3:]
            os.rename(os.path.join(PATH_TO_DATA, directory), os.path.join(PATH_TO_DATA, new_directory))

    def preprocess(self, image: np.array, directory: str) -> np.array:

        """ Preprocess data
        """

        image = cv2.imread(os.path.join(PATH_TO_DATA, directory, image), 1) # 0 = Grayscale, 1 = Color, -1 = Unchanged
        processed_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        return processed_image

    def read_data_from_pickle(self) -> list:

        """ Read image data and labels from pickle file
        """

        labels, data = [], []

        with open("/mnt/e/data/bing_dog_breeds_rgb_150x150.pickle", "rb") as pickle_dataset:
            dataset = pickle.load(pickle_dataset)
        
        for a, b in dataset:
            labels.append(a)
            data.append(b)

        data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        
        # print(labels[-1])
        # img = Image.fromarray(data[10000])
        # img.save("test.png")

        return labels, data

    @staticmethod
    def write_images_from_pickle():

        """ Write image files from pickle
        """

        file_counter = {}

        with open("/mnt/e/data/stanford_dog_breed/stanford_dogs_data_grayscale_categorical.pickle", "rb") as pickle_dataset:
            dataset = pickle.load(pickle_dataset)
        
        for entry in dataset:
            # Try creating folder of class if not exists
            PATH_CLASS = os.path.join(PATH_TO_PREPROCESSED_DATA, entry[0])
            try:
                os.mkdir(PATH_CLASS)
                file_counter[entry[0]] = 1
            except FileExistsError:
                pass
            # Save image in folder of class with id
            cv2.imwrite(os.path.join(PATH_CLASS, str(entry[0]) + "_" + str(file_counter[entry[0]]) + ".jpg"), entry[1])
            # Increment file counter
            file_counter[entry[0]] += 1
        
        print("Files written")


d = Data()
d.write_pickle()