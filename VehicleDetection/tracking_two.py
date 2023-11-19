# Program that performs addvanced counting based on lines and two point of contact, also it finds direction of 'car'
# code commented by harsh
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

area_c = set()

model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('veh2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count = 0
area = [(270, 238), (285, 262), (592, 226), (552, 207)]  # area of interest argument for polylines
tracker = Tracker()

cy1 = 323
cy2 = 367
# y co-ordinates of a line, offset default is 6
offset = 6
vh_down = {}
vh_up = {}
counter = []
counter1 = []

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
        # going down
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = cy  # this saves the car's id:current y position in a dict
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                if counter.count(id) == 0:
                    counter.append(id)
                # this if condition prevents reccuring false count
        # going up
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = cy  # this saves the car's id:current y position in a dict
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                if counter1.count(id) == 0:
                    counter1.append(id)

    cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, ('1line'), (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, ('2line'), (181, 363), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    # print(vh_down)
    l = len(counter)
    cv2.putText(frame, ('Going down - ') + str(l), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    u = len(counter1)
    cv2.putText(frame, ('Going up - ') + str(u), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
