import numpy as np
import cv2
import sys
from os import walk


# wrapper for a functionality that finds the ROI for a face in an image
class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # detect the face in the given image
    # return ROI (cropped image) that defines the face region
    # this ROI can be rotated to make the face appear more upright
    # if there are multiple faces in the image, return the first one
    def detect_face(self, path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angles = [0, 10, -10, 20, -20, 30, -30]
        for a in angles:
            img_rotated = rotate_image(img, a)
            gray_rotated = rotate_image(gray, a)
            faces = self.face_cascade.detectMultiScale(gray_rotated, 1.2, 5)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                roi = img_rotated[int(y):int(y+h), int(x):int(x+w)]
                return roi
        sys.stderr.write("Warning: FaceDetector: no face detected in " + path + "\n")
        return None


# a utility tool used to rotate images to detect angled faces
def rotate_image(image, angle):
    if angle == 0:
        return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result


if __name__ == "__main__":
    fd = FaceDetector()
    path = "./images/set1/"
    _, _, filenames = next(walk(path))
    filenames.sort()
    for f in filenames:
        face = fd.detect_face(path + f)
        if face is None:
            continue
        cv2.imshow("face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

