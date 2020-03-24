import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model


PATH_TO_MODEL = "/mnt/e/data/20200212_model.h5"
# CATEGORIES = [
#     "chihuahua", "japanese_spaniel", "maltese_dog", "pekinese", "shih-tzu", "blenheim_spaniel", "papillon", "toy_terrier", "rhodesian_ridgeback", "afghan_hound", "basset", 
#     "beagle", "bloodhound", "bluetick", "black-and-tan_coonhound", "walker_hound", "english_foxhound", "redbone", "borzoi", "irish_wolfhound", "italian_greyhound", "whippet", 
#     "ibizan_hound", "norwegian_elkhound", "otterhound", "saluki", "scottish_deerhound", "weimaraner", "staffordshire_bullterrier", "american_staffordshire_terrier", "bedlington_terrier", 
#     "border_terrier", "kerry_blue_terrier", "irish_terrier", "norfolk_terrier", "norwich_terrier", "yorkshire_terrier", "wire-haired_fox_terrier", "lakeland_terrier", "sealyham_terrier", 
#     "airedale", "cairn", "australian_terrier", "dandie_dinmont", "boston_bull", "miniature_schnauzer", "giant_schnauzer", "standard_schnauzer", "scotch_terrier", "tibetan_terrier", 
#     "silky_terrier", "soft-coated_wheaten_terrier", "west_highland_white_terrier", "lhasa", "flat-coated_retriever", "curly-coated_retriever", "golden_retriever", "labrador_retriever", 
#     "chesapeake_bay_retriever", "german_short-haired_pointer", "vizsla", "english_setter", "irish_setter", "gordon_setter", "brittany_spaniel", "clumber", "english_springer", 
#     "welsh_springer_spaniel", "cocker_spaniel", "sussex_spaniel", "irish_water_spaniel", "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", "komondor", 
#     "old_english_sheepdog", "shetland_sheepdog", "collie", "border_collie", "bouvier_des_flandres", "rottweiler", "german_shepherd", "doberman", "miniature_pinscher", 
#     "greater_swiss_mountain_dog", "bernese_mountain_dog", "appenzeller", "entlebucher", "boxer", "bull_mastiff", "tibetan_mastiff", "french_bulldog", "great_dane", "saint_bernard", 
#     "eskimo_dog", "malamute", "siberian_husky", "affenpinscher", "basenji", "pug", "leonberg", "newfoundland", "great_pyrenees", "samoyed", "pomeranian", "chow", "keeshond", 
#     "brabancon_griffon", "pembroke", "cardigan", "toy_poodle", "miniature_poodle", "standard_poodle", "mexican_hairless", "dingo", "dhole", "african_hunting_dog"
# ]
CATEGORIES = [
    "chihuahua", "maltese", "afghan", "beagle", "basset", "irish_wolfhound", "whippet", "norwegian_elkhound", "otterhound", "saluki", "scottish_deerhound", "weimaraner", 
    "american_staffordshire_terrier", "bedlington_terrier", "border_terrier", "irish_terrier", "norfolk_terrier", "yorkshire_terrier", "sealyham_terrier", "airedale", "cairn", "boston_bull", 
    "schnauzer", "scottish_terrier", "tibetan_terrier", "lhasa", "golden_retriever", "labrador", "german_short_hair_pointner", "vizla", "english_setter", "irish_setter", "gordon_setter", 
    "brittany_spaniel", "cocker_spaniel", "kuvasz", "schipperke", "collie", "rottweiler", "german_shepherd", "dobermann", "bernese_mountain", "appenzeller", "boxer", "tibetan_mastiff", 
    "great_dane", "saint_bernard", "husky", "affenpinscher", "pug", "leonberg", "newfoundland", "great_pyrenees", "samoyed", "pomeranian", "chow", "keeshond", "pembroke", "cardigan", "poodle", 
    "mexican_hairless", "dingo", "dhole", "african_hunting", "redbone"
]


model = load_model(PATH_TO_MODEL)
model.summary()

image = cv2.imread("/mnt/c/Users/schla/Pictures/human.jpg", 1)
processed_image = cv2.resize(image, (150, 150))
processed_image = np.array(processed_image).reshape(-1, 150, 150, 3)
processed_image = np.array(processed_image, dtype=np.float32)
processed_image /= 255

prediction = model.predict(processed_image)
index = np.argmax(prediction[0])
probability = round(prediction[0][index], 3) * 100
prediction = CATEGORIES[index]

print(index)
print(probability)
print(prediction)
