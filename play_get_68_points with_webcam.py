from imutils import face_utils
import dlib
import numpy as np
import cv2
import os
import face_recognition


def avg(iterable):
    return sum(iterable) / len(iterable)


class SinglePerson:
    def __init__(self, name, max_internal_frames,
                 mouth_window_length, eye_window_length,
                 mouth_close_thermal=None, eye_close_thermal=None,
                 ):
        self.tick = 0
        self.name = name
        self.max_internal_frames = max_internal_frames
        self.mouth_window_length = mouth_window_length
        self.eye_window_length = eye_window_length
        self.eyes_mark_level_window = []
        self.eyes_mark_state_window = []
        self.mouth_mark_level_window = []
        self.mouth_mark_state_window = []
        self.mouth_close_thermal = mouth_close_thermal
        self.eye_close_thermal = eye_close_thermal

    def feed_landmark(self, tick, landmark):
        if tick < self.tick:
            self.reset()
            return
        if tick - self.tick >= self.max_internal_frames:
            self.reset()
            return
        self.process_mouth_landmark(landmark)
        self.process_eyes_landmark(landmark)

    def process_mouth_landmark(self, landmark):
        mouth_close_level = (landmark['top_lip'][-3][1] - landmark['bottom_lip'][-3][1]) / (
                landmark['top_lip'][-5][0] - landmark['top_lip'][-1][0])
        self.mouth_mark_level_window.append(mouth_close_level)
        if self.mouth_close_thermal is not None:
            self.mouth_mark_state_window.append(1 if mouth_close_level < self.mouth_close_thermal else 0)
        if len(self.mouth_mark_level_window) >= self.mouth_window_length:
            avg_level = avg(self.mouth_mark_level_window)
            percent_close = None
            if self.mouth_close_thermal is not None:
                percent_close = avg(self.mouth_mark_state_window)
            self.mouth_mark_state_window = []
            self.mouth_mark_level_window = []
            return avg_level, percent_close
        return None, None

    def process_eyes_landmark(self, landmark):
        eye_close_level = []
        eye_close_state = []
        for each in [landmark['left_eye'], landmark['right_eye']]:
            level = (each[1][1] + each[2][1] - each[4][1] - each[5][1]) / 2 / (each[0][0] - each[3][0])
            eye_close_level.append(level)
            if self.eye_close_thermal is not None:
                eye_close_state.append(eye_close_level < self.eye_close_thermal)
        self.eyes_mark_level_window.append(tuple(eye_close_level))
        if self.eye_close_thermal is not None:
            self.eyes_mark_state_window.append(type(eye_close_state))
        if len(self.eyes_mark_level_window) >= self.eye_window_length:
            left = avg([each[0] for each in self.eyes_mark_level_window])
            right = avg([each[1] for each in self.eyes_mark_level_window])
            avg_eye_close_level = (left, right)
            percent_eye_close = (None, None)
            if self.eye_close_thermal is not None:
                p_left = avg([each[0] for each in self.eyes_mark_state_window])
                p_right = avg([each[1] for each in self.eyes_mark_state_window])
                percent_eye_close = (p_left, p_right)
            self.eyes_mark_level_window = []
            self.eyes_mark_state_window = []
            return avg_eye_close_level, percent_eye_close
        return (None, None), (None, None)

    def reset(self):
        self.eyes_mark_level_window = []
        self.eyes_mark_state_window = []
        self.mouth_mark_level_window = []
        self.mouth_mark_state_window = []


def load_known_face(directory):
    res = []
    for each in os.listdir(directory):
        name = each.split('.')[0]
        face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(f'{directory}/{each}'))[0]
        res.append((name, face_encoding))
    return res


def process_video(video_capture: cv2.VideoCapture):
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:

            rects = face_recognition.face_locations(rgb_small_frame)
            landmarks = face_recognition.face_landmarks(rgb_small_frame)
            print(zip(rects, landmarks))
            for i, (rect, landmark) in enumerate(zip(rects, landmarks)):
                (x, y, w, h) = rect  # 返回人脸框的左上角坐标和矩形框的尺寸
                print(i, rect, landmark)
                # x *= 4
                # y *= 4
                # w *= 4
                # h *= 4
                cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(small_frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for l, points in landmark.items():
                    if 'lip' in l:
                        for j, (x, y) in enumerate(points):
                            # y *= 4
                            cv2.putText(small_frame, "Point #{}".format(j), (x - 10, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.circle(small_frame, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow('Video', small_frame)
            process_this_frame = False
        else:
            process_this_frame = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(r'C:\Users\Administrator\OneDrive - TSCN\cgy\机器学习\人脸识别疲劳检测\c.mp4')
    process_video(video_capture)


if __name__ == '__main__':
    main()
