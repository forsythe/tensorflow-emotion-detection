import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('/home/forsythe/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

cv2.namedWindow("crop", cv2.WINDOW_NORMAL)

z = 7 #zoom factor (lower value is higher zoom)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x+w//z,y+h//z),(x+w-w//z,y+h-h//z),(255,0,0),2)
        roi_gray = gray[y+h//z:y+h-h//z, x+w//z:x+w-w//z]
        #print(type(roi_gray))
        cv2.imshow("crop", cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA))
       
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
