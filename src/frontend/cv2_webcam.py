import sys
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("can not open video file")
    sys.exit()
while True:
    ret, img = cap.read()
    if not ret:
        break
    cv2.imshow("win_img", img)
    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
cap.release()