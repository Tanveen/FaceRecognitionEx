import csv, os, sys
# Import PIL
try:
    from PIL import Image
except ImportError:
    import Image
# Import numpy:
import numpy as np
 # import from Facenet and detect_face
from detect_face import detect_face, bulk_detect_face
from facenet import read_images_from_disk, random_rotate_image, train, crop, flip, load_data, load_image, load_model
from facenet import get_model_filenames

## This is the face recognition module for the RESTful Webservice.
## not sure yet


##WRAPPING THE FACENET MODEL?? WILL IT BE A GOOD IDEA ?? (ABSTRACTION)

class RecognitionInWeb(object):

    def __init__(self, model):
    