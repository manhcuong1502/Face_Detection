import cv2
import numpy as np
import os

face_casecade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

id = input("Nhap ID: ")
name = input("Nhap ten: ")
name = name.replace(' ', '_') + '.' + str(format(int(id), "0>4d"))

sampleNum = 0

while (True):

    ret, frame = cap.read()
    print(ret)
    cv2.flip(frame, 1, frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_casecade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
        if not os.path.exists('dataSet/'+name):
            os.makedirs('dataSet/' + name)
        sampleNum += 1

        # cv2.imwrite('dataSet/User.' + str(id) + '.' + str(sampleNum) + '.jpg', gray[y : y + h, x : x+w])
        cv2.imwrite('dataSet/' + name + '/' + name + '.' + str(format(sampleNum, "0>4d")) + '.jpg', gray[y : y + h, x : x + w])

    cv2.imshow('FRAME', frame)
    cv2.waitKey(1)

    if (sampleNum > 200 or (cv2.waitKey(1) & 0xFF== ord('q'))):
        break
cap.release()
cv2.destroyAllWindows()