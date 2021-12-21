# 이미지펼치기
## 모듈 import
```
import cv2, os
import numpy as np
```
## 원본이미지 불러오기
```
img_path = './images/samples/pexels-photo-276690.jpg'
filename, ext = os.path.splitext(os.path.basename(img_path))
ori_img = cv2.imread(img_path)
src = []  # 마우스 좌표값 입력
```
## 이미지 펼치기
```
# mouse callback handler
def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        img = ori_img.copy()

        src.append([x, y])

        for xx, yy in src:  # 각각의 클릭지점에 점을 생성
            cv2.circle(img, center=(xx, yy), radius=5, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.imshow('img', img)  # 점이 찍힌 이미지 디스플레이

        # perspective transform
        if len(src) == 4:  # 4개의 점이 모두 찍혔으면
            src_np = np.array(src, dtype=np.float32)

            width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
            height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

            dst_np = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)

            result = cv2.warpPerspective(ori_img, M=M, dsize=(int(width), int(height)))

            cv2.imshow('result', result)
            cv2.imwrite('./images/results/%s_result%s' % (filename, ext), result)
```

## 원본이미지를 디스플레이 하고 마우스 콜백 함수 입력
```
# main
cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)

cv2.imshow('img', ori_img)
cv2.waitKey(0)
```