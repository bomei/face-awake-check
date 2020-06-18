from imutils import face_utils
import dlib
import numpy as np
import cv2
import os

FACE_RECOGNITION_MODEL_PATH = './dlib_face_recognition_resnet_model_v1.dat'
SHAPE_PREDICTOR_MODEL_PATH = './shape_predictor_68_face_landmarks.dat'


class FaceImgProcessor:
    def __init__(self,
                 detector=dlib.get_frontal_face_detector(),
                 predictor=dlib.shape_predictor(SHAPE_PREDICTOR_MODEL_PATH),
                 face_rec=dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
                 ):
        self.detector = detector
        self.predictor = predictor
        self.face_rec = face_rec

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

    def get_rects(self, rgb_img):
        return self.detector(rgb_img, 1)

    def get_rects_and_shapes(self, rgb_img):
        rects = self.get_rects(rgb_img)
        shapes = []
        for k, rect in enumerate(rects):
            shapes.append(self.predictor(rgb_img, rect))
        return rects, shapes

    def get_rects_and_shapes_and_face_descriptor(self, rgb_img):
        rects = self.get_rects(rgb_img)
        shapes = []
        fds = []
        for k, rect in enumerate(rects):
            shape = self.predictor(rgb_img, rect)
            shapes.append(shape)
            fd = self.face_rec.compute_face_descriptor(rgb_img, shape)
            fds.append(fd)
        return rects, shapes, fds
