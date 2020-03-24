import os
from icrawler.builtin import BingImageCrawler


PATH_IMAGES = "/mnt/e/data/bing_dog_breeds"
PATH_TO_EXAMPLES = "/mnt/e/data/example_images_65"


class Images():

    def __init__(self):
        
        """ Class init
        """

        pass
    
    @staticmethod
    def download():
        
        """ Download images of every specified breed from Bing
        """

        with open("breeds.csv", "r") as file:
            # Iterate through dog breeds in file
            for entry in file:
                # Fetch variables
                idx, name = entry.split(", ")
                name = name.strip("\n")
                keyword_breed = name + " dog face"
                # Search for keyword and download images
                bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': os.path.join(PATH_IMAGES, idx + "_" + name)})
                bing_crawler.crawl(keyword=keyword_breed, filters=None, offset=0, max_num=400)

    @staticmethod
    def generate_examples():

        """ Delete example images, that are not used in current cnn
        """

        name_list = []

        with open("breeds.csv", "r") as file:
            # Iterate through dog breeds in file
            for entry in file:
                # Fetch variables
                idx, name = entry.split(", ")
                name = name.strip("\n")
                name = name.replace("-", "_")
                image_name = name + ".jpg"
                name_list.append(image_name)

        folder_classes = os.listdir(PATH_TO_EXAMPLES)
        
        for image in folder_classes:
            if image not in name_list:
                os.remove(os.path.join(PATH_TO_EXAMPLES, image))
        
        # Show mission files
        for element in name_list:
            if element not in folder_classes:
                os.mknod(os.path.join(PATH_TO_EXAMPLES, element))


i = Images()
i.generate_examples()
