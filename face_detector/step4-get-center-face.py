import sys
import numpy as np
import cv2
import dlib

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/data/shape_predictor_68_face_landmarks.dat')

# load video
cap = cv2.VideoCapture('assets/mp4/girl.mp4')
# load overlay image
overlay = cv2.imread('assets/images/ryan_transparent.png', cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED: 알파값을 표혀하기 위해


# loop
while True:
    # read frame buffer from video
    ret, img = cap.read()

    if not ret:
        break

    # resize frame (비데오의 크기를 줄인다.)
    scaler = 0.3
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    # detect faces
    faces = detector(img)
    face = faces[0]

    #  현재 이미지에 사각형을 그리는 구문
    img = cv2.rectangle(
        img,
        pt1=(face.left(), face.top()),
        pt2=(face.right(), face.bottom()),
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA

    )

    #  이미지와 얼굴 영역을 입력함
    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    # print(shape_2d)  # [[246 101].. 68개]

    for s in shape_2d:  # 68개의 위치에 점을 찍는다
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 좌상단 우하단 구하고 센터 찾기
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)

    # draw min, max coords
    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # compute face center
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # visualize
    cv2.imshow('original', img)
    # cv2.imshow('facial landmarks', img)
    # cv2.imshow('result', result)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)