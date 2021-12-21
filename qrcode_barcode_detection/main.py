import pyzbar.pyzbar as pyzbar
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('images/samples/bh_bar.jpg')
# img = cv2.imread('images/samples/youtube.jpg')
plt.imshow(img)
# plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')

# plt.show()

decoded = pyzbar.decode(gray)

for d in decoded:
    print(d.data.decode('utf-8'))
    print(d.type)

    cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 0, 255), 2)

plt.imshow(img)
plt.show()

#
## 비데오에서 캡쳐하기
# cap = cv2.VideoCapture(0)
#
# i = 0
# while (cap.isOpened()):
#     ret, img = cap.read()
#
#     if not ret:
#         continue
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     decoded = pyzbar.decode(gray)
#
#     for d in decoded:
#         x, y, w, h = d.rect
#
#         barcode_data = d.data.decode("utf-8")
#         barcode_type = d.type
#
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#         text = '%s (%s)' % (barcode_data, barcode_type)
#         cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#
#     cv2.imshow('img', img)
#
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     elif key == ord('s'):
#         i += 1
#         cv2.imwrite('c_%03d.jpg' % i, img)
#
# cap.release()
# cv2.destroyAllWindows()
#
#
