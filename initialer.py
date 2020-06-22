from imutils import face_utils
import dlib
import numpy as np
import cv2
import os
import face_recognition


class FaceImgProcessor:
    def __init__(self):
        pass

    @staticmethod
    def delta_h_eye(shape):
        left = shape[36][1] + shape[37][1] - shape[39][1] - shape[40][1]
        right = shape[42][1] + shape[43][1] - shape[46][1] - shape[45][1]
        return left / 2, right / 2

    @staticmethod
    def delta_w_eye(shape):
        return shape[38][0] - shape[35][0]

    @staticmethod
    def delta_h_mouth(shape):
        return (shape[60][1] + shape[61][1] + shape[62][1] - shape[64][1] - shape[65][1] - shape[66][1]) / 3

    @staticmethod
    def delta_w_mouth(shape):
        return shape[63][0] - shape[59][0]

    @staticmethod
    def get_locations(image):
        return face_recognition.face_locations(image)

    @staticmethod
    def get_face_landmarks(image):
        return face_recognition.face_landmarks(image)

    @staticmethod
    def get_face_encodings(image):
        return face_recognition.face_encodings(image)

    @staticmethod
    def compare_faces(target_face_list, face):
        return face_recognition.compare_faces(target_face_list, face)
