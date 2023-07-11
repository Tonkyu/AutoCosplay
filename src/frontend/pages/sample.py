import av
import cv2
from PIL import Image
container = av.open("gamer.mp4")
for i, frame in enumerate(container.decode(video=0)):
    img = frame.to_ndarray()
    cv2.imshow("win_img", img)
    print(img.shape)
    cv2.waitKey(0)
    impil = Image.fromarray(img)
    impil.show()
    if i == 0:
        break
"""
cap = cv2.VideoCapture("gamer.mp4")
cnt = 0
while True:
    ret, src = cap.read()
    if not ret:
        break
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if cnt == 0:
        print(dst.shape)
    cv2.imshow("win_dst", dst)
    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
"""