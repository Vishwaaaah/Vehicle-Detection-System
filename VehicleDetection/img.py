# Program that performs counting of 'car' based on pointPolygonTest method
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from datetime import datetime
import os

now = datetime.now()

area_c = set()

model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('vidyolov8.mp4')

my_file = open("C://Users//vishw//Projects//Final_Python_GRP//VehicleDetection//coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count = 0
area = [(270, 238), (285, 262), (592, 226), (552, 207)]  # area of interest argument for polylines
tracker = Tracker()


def imgwrite(img):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = '%s.png' % current_time
    cv2.imwrite(os.path.join(r"C://Users//vishw//Projects//Final_Python_GRP",filename), img)


while True:

    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    # print(results)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    # makes a dataframe from result that has bounding box in 0,1,2,3 column, confidence limit in 4 and class in 5
    # print(px)

    list = []  # contain co-ordinates of bounding box for one object,updated by using update method from tracker
    for index, row in px.iterrows():
        # print(row) #prints rows that are needed to be cleared

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if 'car' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox  # cx,cy are center of rectangle
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)

        if results >= 0:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # -1 draws dot in the center of rectangle
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            crop = frame[y3:y4, x3:x4]
            imgwrite(crop)
            area_c.add(id)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 0), 3)
    count = len(area_c)
    cv2.putText(frame, str(count), (61, 146), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
