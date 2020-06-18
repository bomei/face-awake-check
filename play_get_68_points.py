from imutils import face_utils
import dlib

import cv2

LEFT_EYE_INDEX = range(36, 42)
RIGHT_EYS_INDEX = range(42, 48)


def cal_triangle_area(p1, p2=None, p3=None):
    if isinstance(p1, list):
        if len(p1) == 3:
            (x1, y1), (x2, y2), (x3, y3) = p1
        else:
            return 0
    elif isinstance(p1, list) or isinstance(p1, tuple):
        if p2 is None or p3 is None:
            return 0
        else:
            (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

image = cv2.imread('./face2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

# enumerate()方法用于将一个可遍历的数据对象(列表、元组、字典)组合
# 为一个索引序列，同时列出 数据下标 和 数据 ，一般用在for循环中
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)  # 标记人脸中的68个landmark点
    shape = face_utils.shape_to_np(shape)  # shape转换成68个坐标点矩阵

    (x, y, w, h) = face_utils.rect_to_bb(rect)  # 返回人脸框的左上角坐标和矩形框的尺寸
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    left_eye_points = []
    right_eye_points = []
    for landmark_num in LEFT_EYE_INDEX:
        x, y = shape[landmark_num]
        left_eye_points.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        # cv2.putText(image, "{}".format(landmark_num), (x, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

    area = cal_triangle_area(left_eye_points[:3]) \
           + cal_triangle_area(left_eye_points[1:])
    cv2.putText(image, f'{area}', left_eye_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for landmark_num in RIGHT_EYS_INDEX:
        x, y = shape[landmark_num]
        right_eye_points.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        # cv2.putText(image, "{}".format(landmark_num), (x, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

    area = cal_triangle_area(right_eye_points[:3]) \
           + cal_triangle_area(right_eye_points[1:])
    cv2.putText(image, f'{area}', (right_eye_points[0][0] + 20, right_eye_points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 1)

    # for landmark_num, (x,y) in enumerate(shape):
    #     cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    #     cv2.putText(image, "{}".format(landmark_num), (x, y),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

cv2.imshow("Output", image)
cv2.waitKey(0)
