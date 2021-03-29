import numpy as np
import cv2
from os import walk
from face_detector import FaceDetector


# a wrapper for a utility that detects eyes
class EyeDetector:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # given the face ROI
    # return the (x, y, w, h) bounding box for the eyes
    def detect_eyes(self, roi):
        eyes = self.eye_cascade.detectMultiScale(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        cv2.imshow("eyes", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    path = "./images/set3/"
    _, _, filenames = next(walk(path))
    filenames.sort()
    ed = EyeDetector()
    fd = FaceDetector()
    for f in filenames:
        img = cv2.imread(path + f)
        face, angle = fd.detect_face(img)
        if face is None:
            continue
        ed.detect_eyes(face)
