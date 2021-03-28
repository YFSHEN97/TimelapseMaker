import numpy as np
import cv2
from face_detector import FaceDetector
from os import walk


# wrapper for a utility that returns the apparent age of a still face image
class AgePredictor:
    def __init__(self):
        # age model
        # model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
        # pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
        self.age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")
        self.fd = FaceDetector()

    # given an image
    # extract roi using face detector and predict the age using the age model
    def predict_age(self, path):
        # extract roi and resize it to the desired dimensions for the age model
        roi = self.fd.detect_face(path)
        if roi is None:
            return -1
        roi_resized = cv2.resize(roi, (224, 224))
        img_blob = cv2.dnn.blobFromImage(roi_resized)
        # run it through the model and return predicted age
        self.age_model.setInput(img_blob)
        age_dist = self.age_model.forward()[0]
        output_indexes = np.array([i for i in range(0, 101)])
        apparent_age = round(np.sum(age_dist * output_indexes), 2)
        return apparent_age


if __name__ == "__main__":
    ap = AgePredictor()
    path = "./images/set3/"
    _, _, filenames = next(walk(path))
    filenames.sort()
    for f in filenames:
        age = ap.predict_age(path + f)
        print(f + ": " + str(age))
