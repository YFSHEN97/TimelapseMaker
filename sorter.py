import cv2
import numpy as np
from age_predictor import AgePredictor
from os import walk


class Sorter:
    # path specifies a directory that contains all the face images to be sorted
    # only accepts JPEG files!!
    # these images will be read as numpy arrays and stored in a list
    def __init__(self, path):
        # a list of triples: [path+filename, color_image, predicted_age]
        self.images = list()
        # age predictor
        self.ap = AgePredictor()
        # walk the given path directory and read the images
        _, _, filenames = next(walk(path))
        filenames.sort()
        for f in filenames:
            if f[-5:] != ".jpeg":
                continue
            self.images.append([path + f, cv2.imread(path + f), -1])


    # sorts the set of images based on age
    def sort(self):
        for image in self.images:
            path = image[0]
            img = image[1]
            image[2] = self.ap.predict_age(path)
        self.images.sort(key=lambda i: i[2])
        return self.images


if __name__ == "__main__":
    sorter = Sorter("./images/set1/")
    images_sorted = sorter.sort()
    for i in images_sorted:
        winname = i[0] + ": " + str(i[2]) + " years old"
        h, w, _ = i[1].shape
        sf = 400 / h
        dsize = (int(w * sf), 400)
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 40,30)
        cv2.imshow(winname, cv2.resize(i[1], dsize))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
