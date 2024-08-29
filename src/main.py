from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

cap = cv2.VideoCapture("../assets/Videos/trafic_camera.mp4")  # For Video
model = YOLO("../model/yolov8s.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

mask = cv2.imread("../assets/mask.png")

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsUp = [250, 400, 575, 400]
limitsDown = [700, 450, 1125, 450]
total_countsUp = {"car":[], "bus":[], "truck":[]}
total_countsDown = {"car":[], "bus":[], "truck":[]}

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img , mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            if currentclass == 'car' or currentclass == 'bus'\
                  or currentclass == 'truck' or currentclass == 'motorbike'\
                    and conf > 0.4:
                cvzone.cornerRect(img, (x1, y1, w, h) , l=15)
                cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)),
                                    scale=2,
                                    thickness=2,
                                    offset=3)
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), color=(0, 0, 0), thickness= 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), color=(0, 0, 0), thickness= 5)

    resultracker = tracker.update(detections)
    for results in resultracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1+w//2 , y1+h//2
        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-15 < cy < limitsUp[3]+15:
            total_countsUp[str(classNames[cls])].append(id)
        cvzone.putTextRect(img, f'car_counts  :{len(set(total_countsUp["car"]))}', (50,50),
                            scale=2,
                            thickness=1,
                            offset=3, colorT=(0,0,0))
        cvzone.putTextRect(img, f'bus_counts  :{len(set(total_countsUp["bus"]))}', (50,80),
                            scale=2,
                            thickness=1,
                            offset=3, colorT=(0,0,0))
        cvzone.putTextRect(img, f'truck_counts:{len(set(total_countsUp["truck"]))}', (50,110),
                            scale=2,
                            thickness=1,
                            offset=3, colorT=(0,0,0))

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-15 < cy < limitsDown[3]+15:
            total_countsDown[str(classNames[cls])].append(id)
        cvzone.putTextRect(img, f'car_counts  :{len(set(total_countsDown["car"]))}', (1000,50),
                            scale=2,
                            thickness=1,
                            offset=3, colorT=(0,0,0))
        cvzone.putTextRect(img, f'bus_counts  :{len(set(total_countsDown["bus"]))}', (1000,80),
                            scale=2,
                            thickness=1,
                            offset=3, colorT=(0,0,0))
        cvzone.putTextRect(img, f'truck_counts:{len(set(total_countsDown["truck"]))}', (1000,110),
                            scale=2,
                            thickness=1,
                            offset=3, colorT=(0,0,0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)