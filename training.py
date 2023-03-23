import cv2
import numpy as np
import sqlite3
import os
from PIL import Image


def insertOrUpdate(id, name):
    connect = sqlite3.connect('./data.db')
    query = "SELECT * FROM people WHERE ID=" + str(id)

    cursor = connect.execute(query)

    isRecordExist = 0

    for row in cursor:
        isRecordExist = 1

    if(isRecordExist == 0):
        query = "INSERT INTO people(ID, name) VALUES(" + str(id) + ",'" + str (name) + "')"
    else:
        query = "UPDATE people SET name ='" + str(name) + "' WHERE ID=" + str(id)

    connect.execute(query)

    connect.commit()
    connect.close()

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'dataSet'

def getImage_ID(path):
    folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for folderPath in folderPaths:
        imagePaths = [os.path.join(folderPath, f) for f in os.listdir(folderPath)]
        db = folderPath.split('\\')[-1].split('.')
        name = db[0].replace('_', ' ')
        id = db[1]
        print(id)
        print(name)
        insertOrUpdate(id, name)
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')

            faces.append(faceNp)
            IDs.append(int(id))
    return faces, IDs

faces, IDs = getImage_ID(path)
recognizer.train(faces, np.array(IDs))
recognizer.save('./model/trainingData.yml')
cv2.destroyAllWindows()