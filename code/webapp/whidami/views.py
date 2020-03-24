import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model

from django.template import loader
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

from .forms import FileForm


CATEGORIES = [
    "chihuahua", "maltese", "afghan", "beagle", "basset", "irish_wolfhound", "whippet", "norwegian_elkhound", "otterhound", "saluki", "scottish_deerhound", "weimaraner", 
    "american_staffordshire_terrier", "bedlington_terrier", "border_terrier", "irish_terrier", "norfolk_terrier", "yorkshire_terrier", "sealyham_terrier", "airedale", "cairn", "boston_bull", 
    "schnauzer", "scottish_terrier", "tibetan_terrier", "lhasa", "golden_retriever", "labrador", "german_short_hair_pointner", "vizla", "english_setter", "irish_setter", "gordon_setter", 
    "brittany_spaniel", "cocker_spaniel", "kuvasz", "schipperke", "collie", "rottweiler", "german_shepherd", "dobermann", "bernese_mountain", "appenzeller", "boxer", "tibetan_mastiff", 
    "great_dane", "saint_bernard", "husky", "affenpinscher", "pug", "leonberg", "newfoundland", "great_pyrenees", "samoyed", "pomeranian", "chow", "keeshond", "pembroke", "cardigan", "poodle", 
    "mexican_hairless", "dingo", "dhole", "african_hunting", "redbone"
]


def homepage_view(request):
    
    """ Function to render homepage
    """
    
    probability, breed, image_path_dog = "", "", ""

    # Load template
    template_home = loader.get_template("index.html")

    if request.method == "POST":

        # Load template
        template_result = loader.get_template("result.html")
        # Create file form
        form = FileForm(request.POST, request.FILES)

        if form.is_valid():
            
            # Check if file is valid
            form.save()
            uploaded_file = request.FILES["file"].read()

            # Read image from file and resize
            image = cv2.imread("media/files/" + str(request.FILES["file"]), 1)
            processed_image = cv2.resize(image, (150, 150))
            processed_image = np.array(processed_image).reshape(-1, 150, 150, 3)
            processed_image = np.array(processed_image, dtype=np.float32)
            processed_image /= 255

            # Load model (tensorflow 2.1.0 needed!)
            model = load_model("media/model.h5")
            # Predict current image
            prediction = model.predict(processed_image)
            # Get class and probability
            index = np.argmax(prediction[0])
            probability = round(float(prediction[0][index]), 2) * 100
            # Get breed name
            breed = CATEGORIES[index]
            image_path_dog = breed + ".jpg"
            breed = breed.replace("_", " ")

            # Fetch path to example dog breed image

            context = {
                "probability": probability,
                "breed": breed,
                "image_path_human": str(request.FILES["file"]),
                "image_path_dog": image_path_dog,
            }

            return HttpResponse(template_result.render(context, request))
        
        else:
            # Reset form if file not valid
            form = FileForm()

    # Context to fill template
    context = {}

    return HttpResponse(template_home.render(context, request))


def result_view(request):

    """ Function to render homepage
    """

    probability, breed = "", ""
    image_path_human = ""
    image_path_dog = ""

    
    # Load html template
    template = loader.get_template("result.html")

    # Context to fill template
    context = {
        "probability": probability,
        "breed": breed,
        "image_path_human": image_path_human,
        "image_path_dog": image_path_dog,
    }

    if "back" in request.POST:
        
        # Return to homepage if button is pressed
        return HttpResponseRedirect("/")

    return HttpResponse(template.render(context, request))


def about_view(request):

    """ Function to render about page
    """

    return render(request, "about.html", {})


def imprint_view(request):

    """ Function to render imprint page
    """
  
    return render(request, "imprint.html", {})


def privacy_view(request):

    """ Function to render privacy page
    """
  
    return render(request, "privacy.html", {})
