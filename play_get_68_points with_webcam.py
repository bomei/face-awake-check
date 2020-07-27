from imutils import face_utils
import numpy as np
import cv2
import os
import face_recognition
import logging as syslog
from bo_log import log, set_log
from typing import Dict
import arrow
from dlib_test import dlib_process_img

set_log(syslog.DEBUG)


def avg(iterable):
    return sum(iterable) / len(iterable)


class SinglePerson:
    def __init__(self, name, max_internal_frames=25 * 60,
                 mouth_window_length=25 * 60, eye_window_length=25 * 30,
                 mouth_close_thermal=0.3, eye_close_thermal=0.24,
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
        mouth_avg, mouth_percent = self.process_mouth_landmark(landmark)
        (left_eye_avg, right_eye_avg), (left_eye_percent, right_eye_percent) = self.process_eyes_landmark(landmark)
        self.tick = tick
        results = [mouth_avg, mouth_percent, left_eye_avg, left_eye_percent, right_eye_avg, right_eye_percent]
        if any(results):
            return results
        else:
            return None

    def process_mouth_landmark(self, landmark):
        # TODO: logic not right here
        mouth_close_level = (landmark['top_lip'][-3][1] - landmark['bottom_lip'][-3][1]) / (
                landmark['top_lip'][-5][0] - landmark['top_lip'][-1][0])
        self.mouth_mark_level_window.append(mouth_close_level)
        if self.mouth_close_thermal is not None:
            self.mouth_mark_state_window.append(1 if mouth_close_level > self.mouth_close_thermal else 0)
        if len(self.mouth_mark_level_window) >= self.mouth_window_length:
            avg_level = avg(self.mouth_mark_level_window)
            percent_close = None
            if self.mouth_close_thermal is not None:
                percent_close = avg(self.mouth_mark_state_window)
            self.mouth_mark_state_window = []
            self.mouth_mark_level_window = []
            log.debug('process mouth', avg_mouth_level=avg_level, percent_mouth_close=percent_close, self_tick=self.tick
                      )
            return avg_level, percent_close
        return None, None

    def process_eyes_landmark(self, landmark):
        eye_close_level = []
        eye_close_state = []
        for each in [landmark['left_eye'], landmark['right_eye']]:
            level = (each[1][1] + each[2][1] - each[4][1] - each[5][1]) / 2 / (each[0][0] - each[3][0])
            eye_close_level.append(level)
            if self.eye_close_thermal is not None:
                eye_close_state.append(level < self.eye_close_thermal)
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
            log.debug('process eyes landmark', avg_eye_close_level=avg_eye_close_level,
                      percent_eye_close=percent_eye_close, self_tick=self.tick)
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


person_dict = {}  # type: Dict[str,SinglePerson]
known_face_pairs = load_known_face(r'C:\Users\zannb\Desktop\face_img\known_faces')
known_face_encodings = [each[1] for each in known_face_pairs]


def process_rgb_frame(raw_frame, rgb_frame, tick):
    rects = face_recognition.face_locations(rgb_frame, model='cnn')
    landmarks = face_recognition.face_landmarks(rgb_frame)
    # TODO:
    #  the landmarks are calculated in face_encodings() already, so here is a duplicated statement.
    #  Have to hack the face_encodings function
    face_encodings = face_recognition.face_encodings(rgb_frame)
    return face_encodings
    for i, (rect, landmark, face_encoding) in enumerate(zip(rects, landmarks, face_encodings)):
        (top, right, bottom, left) = rect  # 返回人脸框的左上角坐标和矩形框的尺寸
        # print(i, rect, landmark)
        # x *= 4
        # y *= 4
        # w *= 4
        # h *= 4
        # log.debug('rect locations', top=top, right=right, bottom=bottom, left=left)
        cv2.rectangle(raw_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        match_result = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = None
        if True in match_result:
            index = match_result.index(True)
            name = known_face_pairs[index][0]
            if name not in person_dict.keys():
                person_dict[name] = SinglePerson(name)
            processed_results = person_dict[name].feed_landmark(tick, landmark)
            if processed_results is not None:
                log.debug('', tick=tick, processed_results=processed_results)
        cv2.putText(raw_frame, f"Face #{name if name is not None else 'unknown'}", (left - 10, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # for l, points in landmark.items():
        #     if 'lip' in l:
        #         for j, (x, y) in enumerate(points):
        #             # y *= 4
        #             cv2.putText(raw_frame, "Point #{}".format(j), (x - 10, y - 10),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #             cv2.circle(raw_frame, (x, y), 5, (255, 0, 0), -1)
    return raw_frame


def process_video(video_capture: cv2.VideoCapture):
    process_this_frame = True
    tick = 0

    while True:
        ret, frame = video_capture.read()
        if frame is None:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if process_this_frame:
            rgb_small_frame = small_frame[:, :, ::-1]
            # small_frame = process_rgb_frame(small_frame, rgb_small_frame, tick)
            # cv2.imshow('Video', small_frame)
            t1 = arrow.now()
            # r1 = np.array(process_rgb_frame(small_frame, rgb_small_frame, 0))
            t2 = arrow.now()
            r2 = np.array(dlib_process_img(small_frame, rgb_small_frame))
            t3 = arrow.now()
            log.debug( fr_time=t2 - t1, dlib_time=t3 - t2)
            process_this_frame = False
            tick += 1
        else:
            process_this_frame = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(r'D:\OneDrive - TSCN\cgy\机器学习\人脸识别疲劳检测\c.mp4')
    process_video(video_capture)


if __name__ == '__main__':
    main()
