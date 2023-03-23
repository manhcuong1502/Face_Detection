import time
import cv2
import sqlite3


face_casecade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./model/trainingData.yml')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./Sample/output1.avi', fourcc, 8.0, (640, 480))

def getProfile(id):
    connect = sqlite3.connect('./data.db')
    query = "SELECT * FROM people WHERE ID=" + str(id)
    cursor = connect.execute(query)

    profile = None

    for row in cursor:
        profile = row

    connect.close()
    return profile

cap = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
row = 2
i = 0
check = 0
clock = time.localtime()

while(True):
    ret, frame = cap.read()

    cv2.flip(frame, 1, frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_casecade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)

        roi_gray = gray[y: y + h, x: x + w]

        id, confidence = recognizer.predict(roi_gray)

        if (confidence < 50):
            profile = getProfile(id)
            if (profile != None):
                cv2.putText(frame, "" + str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                check += 1
        else:
            cv2.putText(frame, "UNKNOWN", (x + 10, y + h + 30), fontface, 1, (0, 0, 225), 2)
            check = 0 

    cv2.imshow( 'CHECK FACE', frame)

    if (check >= 60):
        break
    if (cv2.waitKey(1) == ord('q')):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
print ('Success')