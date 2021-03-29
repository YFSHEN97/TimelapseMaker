import numpy as np
import cv2
from sorter import Sorter
from detect_landmarks import FaceLandmarkDetector
from warp_image import ImageWarper
import extrapolate_vector_field as evf


# given a list of face images already sorted according to age 
# return a new list of these faces, where each has been transformed
# the transformation performs the following steps in order:
### rotation, so that eyes in each image is horizontal
### resizing, so that all images have the same resolution
### scaling warp, so that the two eyes have the same distance in each face
### translation, so that the eyes are aligned in all images
def align_faces(faces):
    results = [None for _ in faces]
    # resize all images so that they are 800*800
    for i in range(len(faces)):
        face = faces[i]
        img = face[1]
        h, w, cc = img.shape
        sf = 800 / max(h, w)
        hh, ww = int(h * sf), int(w * sf)
        img = cv2.resize(img, (ww, hh))
        results[i] = np.full((800, 800, cc), (0,0,0), dtype=np.uint8)
        xx = (800 - ww) // 2
        yy = (800 - hh) // 2
        results[i][yy:yy+hh, xx:xx+ww] = img
    return results


# given a list of transformed faces
# generate a timelapse video out of it
# the previous face is morphed into the next
# interval is the time (in seconds) between each successive face
# pause is the time (in seconds) that we dwell on each face
# fps is the frame rate of the video
def make_video(faces, out_filename, interval=1, pause=0.5, fps=30):
    assert len(faces) > 1
    d = FaceLandmarkDetector()
    e = evf.Extrapolator()
    w = ImageWarper()
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800))
    for i in range(len(faces) - 1):
        face1 = faces[i]
        face2 = faces[i+1]
        landmarks1 = d.predict(face1)
        landmarks2 = d.predict(face2)
        # compute the warping field face1 -> face2
        face1_x = landmarks2[:, 0]
        face1_y = landmarks2[:, 1]
        face1_dx = landmarks1[:, 0] - landmarks2[:, 0]
        face1_dy = landmarks1[:, 1] - landmarks2[:, 1]
        face1_fx, face1_fy = e.extrapolate(face1_x, face1_y, face1_dx, face1_dy, (800,800))
        # compute the warping field face2 -> face1
        face2_x = landmarks1[:, 0]
        face2_y = landmarks1[:, 1]
        face2_dx = landmarks2[:, 0] - landmarks1[:, 0]
        face2_dy = landmarks2[:, 1] - landmarks1[:, 1]
        face2_fx, face2_fy = e.extrapolate(face2_x, face2_y, face2_dx, face2_dy, (800,800))
        # first put original face1 into the video for duration "pause"
        for i in range(int(pause * fps)):
            out.write(face1)
        # then produce the warped sequence
        warp_amounts = np.linspace(0., 1., int(interval * fps))
        for i, warp_amount in enumerate(warp_amounts):
            face1_warped = w.warp(face1, face1_fx, face1_fy, warp_amount)
            face2_warped = w.warp(face2, face2_fx, face2_fy, 1 - warp_amount)
            # We alpha blend the original images
            face_out = (1 - warp_amount) * face1_warped + warp_amount * face2_warped
            # write video frame
            out.write(face_out.astype(np.uint8))
    # put the last face into the video for duration "pause"
    for i in range(int(pause * fps)):
        out.write(faces[-1])
    out.release()


if __name__ == "__main__":
    path = "./images/set3/"
    sorter = Sorter(path)
    sorter.sort()
    facelist = sorter.list_all()
    facelist = align_faces(facelist)
    make_video(facelist, "./data/out.mp4")

