from imutils import face_utils
import dlib
import numpy as np
import cv2
import os

LEFT_EYE = [37, 38, 41, 40]
RIGHT_EYS = [43, 44, 47, 46]

FACE_RECOGNITION_MODEL_PATH = './dlib_face_recognition_resnet_model_v1.dat'
SHAPE_PREDICTOR_MODEL_PATH = './shape_predictor_68_face_landmarks.dat'
KNOWN_FACE_DIR = './known_face'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL_PATH)
face_rec = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)


def delta_h_eye(shape):
    left = shape[36][1] + shape[37][1] - shape[39][1] - shape[40][1]
    right = shape[42][1] + shape[43][1] - shape[46][1] - shape[45][1]
    return left / 2, right / 2


def delta_w_eye(shape):
    return shape[38][0] - shape[35][0]


def delta_h_mouth(shape):
    return (shape[60][1] + shape[61][1] + shape[62][1] - shape[64][1] - shape[65][1] - shape[66][1]) / 3


def delta_w_mouth(shape):
    return shape[63][0] - shape[59][0]


def load_known_faces():  # ->type: dict
    results = {}
    for each in os.listdir(KNOWN_FACE_DIR):
        img = dlib.load_rgb_image(f'{KNOWN_FACE_DIR}/{each}')
        fd = get_face_descriptor(img)
        results[each] = fd
    return results


def get_face_descriptor(img):
    dets = detector(img, 1)
    res = []
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        res.append(face_descriptor)
    return res


def main():
    video_capture = cv2.VideoCapture(0)

    process_this_frame = True

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            rects = detector(rgb_small_frame, 1)
            log_msg = []
            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)  # 返回人脸框的左上角坐标和矩形框的尺寸
                x *= 4
                y *= 4
                w *= 4
                h *= 4
                log_msg.append(f' Face #{i + 1}: ({x},{y})')
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print('|'.join(log_msg))
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def test():
    img = dlib.load_rgb_image(f'./face2.jpg')
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    dets = detector(img,1)

    for k,d in enumerate(dets):
        shape = predictor(img, d)
        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        print(shape)
        print(face_descriptor)


if __name__ == '__main__':
    test()