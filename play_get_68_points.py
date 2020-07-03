from imutils import face_utils
import dlib

import cv2

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(r'D:\OneDrive - TSCN\software\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(
    r'D:\OneDrive - TSCN\cgy\机器学习\人脸识别疲劳检测\code\dlib_face_recognition_resnet_model_v1.dat')


def dlib_process_img(raw_frame, rgb_frame):
    rects = detector(rgb_frame)
    for i, rect in enumerate(rects):
        landmark = shape_predictor(rgb_frame, rect)
        encoding = facerec.compute_face_description(rgb_frame, landmark)
    return encoding
