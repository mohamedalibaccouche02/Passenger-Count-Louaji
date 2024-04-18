import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import *
from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB Atlas
client = MongoClient('mongodb+srv://mohamedalibaccouche:baccouche@cluster0.tqytr4u.mongodb.net/Louaji?retryWrites=true&w=majority')
db = client['Louaji']
collection = db['passengers']

model = YOLO('best.pt')

tracker = Tracker()

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('salem.MOV')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
area1 = [(765, 213), (246, 213), (246, 430),(786, 430)]

pin = {}
enter = []

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    faces_count = 0  # Initialize face count for this frame

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        list.append([x1, y1, x2, y2])
        faces_count += 1  # Increment face count for each detected face
    
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        w, h = x4 - x3, y4 - y3
        result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result >= 0:
            pin[id] = (cx, cy)
            cvzone.cornerRect(frame, (x3, y3, w, h), 10, 5)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            if id not in enter:
                enter.append(id)

    ep = len(enter)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    cvzone.putTextRect(frame, f'counter:{ep}', (50, 60), 2, 2)

    # Display frame
    cv2.imshow("RGB", frame)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Store data in MongoDB Atlas
    doc = {
        "face_count": faces_count,
        "createdAt": timestamp,  # Current time in milliseconds
        "updatedAt":timestamp
    }
    collection.insert_one(doc)

    if cv2.waitKey(2) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
