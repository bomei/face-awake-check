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
    def delta_h_eye(landmark):
        pass

    @staticmethod
    def delta_w_eye(landmark):
        pass

    @staticmethod
    def delta_h_mouth(landmark):
        pass

    @staticmethod
    def delta_w_mouth(landmark):
        pass

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
