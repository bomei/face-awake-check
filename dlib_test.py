from imutils import face_utils
import dlib
import os

import cv2

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("C:\\Users\\zannb\\Desktop\\face-awake-check\\model\\shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("C:\\Users\\zannb\\Desktop\\face-awake-check\\model\\dlib_face_recognition_resnet_model_v1.dat")


def dlib_process_img(raw_frame, rgb_frame):
    rects = detector(rgb_frame,0)
    for i, rect in enumerate(rects):
        landmark = shape_predictor(rgb_frame, rect)
        encoding = facerec.compute_face_descriptor(rgb_frame, landmark)
    return encoding
