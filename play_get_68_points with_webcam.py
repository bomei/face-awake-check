from imutils import face_utils
import dlib
import numpy as np
import cv2
import os
from initialer import FaceImgProcessor


def main():
    fip = FaceImgProcessor()
    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(r'C:\Users\Administrator\OneDrive - TSCN\cgy\机器学习\人脸识别疲劳检测\a.mp4')
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:

            rects = fip.get_locations(rgb_small_frame)
            landmarks = fip.get_face_landmarks(rgb_small_frame)
            print(zip(rects,landmarks))
            for i, (rect, landmark) in enumerate(zip(rects, landmarks)):
                (x, y, w, h) = face_utils.rect_to_bb(rect)  # 返回人脸框的左上角坐标和矩形框的尺寸
                print(i,rect,landmark)
                # x *= 4
                # y *= 4
                # w *= 4
                # h *= 4
                cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(small_frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for (x, y) in landmark:
                    # x *= 4
                    # y *= 4
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        cv2.imshow('Video', small_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
