import sys
import time
import cv2
import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QDir

from demoBase import Ui_Dialog  # Assuming Ui_Dialog is defined in demoBase

# Importing required modules from moviepy, ultralytics, sort, and cvzone
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
from sort import *
import cvzone
import csv


class VideoPlayer(QDialog, Ui_Dialog):

    def __init__(self, *args, obj=None, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.image_size = [self.image.geometry().width(), self.image.geometry().height()]

        # Initialize YOLO model
        self.model = YOLO("../model/best.pt")
        self.classNames = ['truck', 'car', 'minibus', 'bus']

        # Initialize dictionaries to hold QLabel objects
        self.car = {}
        self.carcnt = {}
        self.minibus = {}
        self.minibuscnt = {}
        self.bus = {}
        self.buscnt = {}
        self.truck = {}
        self.truckcnt = {}
        self.sum = {}
        self.dir = {}

        self.connectsignal()

        self.running_flag = 0
        self.pause_flag = 0
        self.get_points_flag = 0
        self.counter_thread_start_flag = 0

        self.pntnum = 0
        self.pointarray = []
        self.linearray = []

        # Flag for starting line
        self.temp = {}
        self.video_paused = False

    def connectsignal(self):
        self.open_button.clicked.connect(self.open_file)
        self.start_button.clicked.connect(self.start_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.line_button.clicked.connect(self.process)
        self.expert_button.clicked.connect(self.export)

        self.image.mouseDoubleClickEvent = self.get_points

    def closeEvent(self, event):
        # Release the video capture when the window is closed
        self.release_video_capture()
        event.accept()

    def release_video_capture(self):
        # Release the video capture if it exists
        if hasattr(self, 'video_capture') and self.cap.isOpened():
            self.video_capture.release()

    def changesignal(self, cnt):
        self.car.clear()
        self.carcnt.clear()
        self.minibus.clear()
        self.minibuscnt.clear()
        self.bus.clear()
        self.buscnt.clear()
        self.truck.clear()
        self.truckcnt.clear()
        self.sum.clear()
        self.dir.clear()
        self.temp.clear()
        signalcnt = 0

        for i in range(cnt):
            self.car[i] = {}
            self.carcnt[i] = {}
            self.minibus[i] = {}
            self.minibuscnt[i] = {}
            self.bus[i] = {}
            self.buscnt[i] = {}
            self.truck[i] = {}
            self.truckcnt[i] = {}
            self.sum[i] = {}
            self.dir[i] = {}
            self.temp[i] = []

            for j in range(cnt):
                self.carcnt[i][j] = 0
                self.minibuscnt[i][j] = 0
                self.buscnt[i][j] = 0
                self.truckcnt[i][j] = 0

                if j == i:
                    continue

                self.dir[i][j] = QLabel("", self.count)
                self.dir[i][j].setGeometry(20, 100 + 40 * signalcnt, 50, 20)
                self.dir[i][j].setStyleSheet("background:None")
                self.dir[i][j].setAlignment(QtCore.Qt.AlignCenter)
                self.dir[i][j].setText(str(str(i) + "->" + str(j)))
                self.dir[i][j].show()

                self.car[i][j] = QLabel("0", self.count)
                self.car[i][j].setGeometry(90, 100 + 40 * signalcnt, 50, 20)
                self.car[i][j].setStyleSheet("background:None")
                self.car[i][j].setAlignment(QtCore.Qt.AlignCenter)
                self.car[i][j].show()

                self.minibus[i][j] = QLabel("0", self.count)
                self.minibus[i][j].setGeometry(180, 100 + 40 * signalcnt, 50, 20)
                self.minibus[i][j].setStyleSheet("background:None")
                self.minibus[i][j].setAlignment(QtCore.Qt.AlignCenter)
                self.minibus[i][j].show()

                self.bus[i][j] = QLabel("0", self.count)
                self.bus[i][j].setGeometry(270, 100 + 40 * signalcnt, 50, 20)
                self.bus[i][j].setStyleSheet("background:None")
                self.bus[i][j].setAlignment(QtCore.Qt.AlignCenter)
                self.bus[i][j].show()

                self.truck[i][j] = QLabel("0", self.count)
                self.truck[i][j].setGeometry(360, 100 + 40 * signalcnt, 50, 20)
                self.truck[i][j].setStyleSheet("background:None")
                self.truck[i][j].setAlignment(QtCore.Qt.AlignCenter)
                self.truck[i][j].show()

                self.sum[i][j] = QLabel("0", self.count)
                self.sum[i][j].setGeometry(QtCore.QRect(450, 100 + 40 * signalcnt, 50, 20))
                self.sum[i][j].setStyleSheet("background:None")
                self.sum[i][j].setAlignment(QtCore.Qt.AlignCenter)
                self.sum[i][j].show()

                signalcnt += 1

    def open_file(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie", QDir.homePath())

        cap = cv2.VideoCapture(self.fileName)
        while cap.isOpened():
            ret, self.frame = cap.read()
            if ret:
                self.show_image(self.frame)
                self.imgScale = np.array(self.frame.shape[:2]) / [self.image_size[1], self.image_size[0]]
                cap.release()
                break

    def show_image(self, img_np):
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, self.image_size)
        bytesPerLine = 3 * self.image_size[0]
        frame = QImage(img_np.data, self.image_size[0], self.image_size[1], bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        self.image.setPixmap(pix)
        self.image.repaint()

    def start_video(self):
        self.cnt = int(self.pntnum / 2)
        self.changesignal(self.cnt)
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        limitsUp = [250, 400, 575, 400]
        limitsDown = [700, 450, 1125, 450]
        total_countsUp = {"car": [], "bus": [], "truck": []}
        total_countsDown = {"car": [], "bus": [], "truck": []}

        if self.fileName is not None:
            self.cap = cv2.VideoCapture(self.fileName)
            new_frame_time = 0

            while True:
                if self.video_paused:
                    # If video is paused, wait for 100 milliseconds before continuing
                    cv2.waitKey(100)
                    continue
                new_frame_time = time.time()
                success, img = self.cap.read()
                for i in range(0, len(self.linearray)):
                    cv2.circle(self.frame, (self.linearray[i][0], self.linearray[i][1]), 5, (255, 0, 0), -1)
                    cv2.circle(self.frame, (self.linearray[i][2], self.linearray[i][3]), 5, (255, 0, 0), -1)

                results = self.model(img, stream=True)
                detections = np.empty((0, 5))
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        currentclass = self.classNames[cls]
                        if currentclass == 'car' or currentclass == 'bus' \
                                or currentclass == 'truck' or currentclass == 'minibus' \
                                and conf > 0.4:
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15)
                            cvzone.putTextRect(img, f'{self.classNames[cls]}', (max(0, x1), max(35, y1)),
                                               scale=2,
                                               thickness=2,
                                               offset=3)
                            currentarray = np.array([x1, y1, x2, y2, conf])
                            detections = np.vstack((detections, currentarray))

                result_tracker = tracker.update(detections)
                for result in result_tracker:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w // 2, y1 + h // 2
                    self.count_vehicle(cx, cy, int(id), cls)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    cv2.putText(img, f'{int(id)}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                self.show_image(img)
                cv2.waitKey(1)
                cv2.destroyAllWindows()

    def condition(self, cx, cy, i):
        [x1, y1, x2, y2] = self.linearray[i]
        minx = min(x1, x2)
        maxx = max(x1, x2)
        miny = min(y1, y2)
        maxy = max(y1, y2)
        if cx > minx and cx < maxx and cy > miny and cy < maxy:
            return True
        else:
            return False

    def count_vehicle(self, cx, cy, num, cls):
        for i in range(self.cnt):

            print(str(self.temp[i]) + "\n")
            if self.condition(cx, cy, i) == False:
                continue
            if i == 0:
                print("OKKKKKKKK")
            print(str(str(num) + "th track is in the ") + str(i) + "th line.")
            if num in self.temp[i]:
                print(str(str(num) + "th track in in temp[" + str(i) + "] already"))
                continue
            flag = 0
            for j in range(self.cnt):
                if j == i:
                    continue
                if num in self.temp[j]:
                    print(str(str(num) + "th track is " + str(j) + "->" + str(i)))
                    flag = 1
                    self.temp[j].remove(num)
                    if cls == 1:
                        self.carcnt[j][i] += 1
                        self.car[j][i].setText(str(self.carcnt[j][i]))
                    elif cls == 2:
                        self.minibuscnt[j][i] += 1
                        self.minibus[j][i].setText(str(self.minibuscnt[j][i]))
                    elif cls == 3:
                        self.buscnt[j][i] += 1
                        self.bus[j][i].setText(str(self.buscnt[j][i]))
                    elif cls == 0:
                        self.truckcnt[j][i] += 1
                        self.truck[j][i].setText(str(self.truckcnt[j][i]))
                    sumcnt = self.carcnt[j][i] + self.minibuscnt[j][i] + self.buscnt[j][i] + self.truckcnt[j][i]
                    self.sum[j][i].setText(str(sumcnt))
                    
                    break
            if flag == 0:
                self.temp[i].append(num)

    def pause_video(self):
        self.video_paused = not self.video_paused

    def get_points(self, event):
        if not self.get_points_flag:
            return
        self.pntnum = self.pntnum + 1
        x = event.x()
        y = event.y()
        realx = int(x * self.imgScale[1])
        realy = int(y * self.imgScale[0])
        cv2.circle(self.frame, (realx, realy), 5, (255, 0, 0), -1)
        self.show_image(self.frame)

        if self.pntnum % 2 == 0 and self.pntnum > 0:
            cv2.line(self.frame, (self.pointarray[-1][0], self.pointarray[-1][1]), (realx, realy), (255, 0, 0), 2)
            self.show_image(self.frame)
            self.linearray.append([self.pointarray[-1][0], self.pointarray[-1][1], realx, realy])
        self.pointarray.append([realx, realy])

    def process(self):
        if self.counter_thread_start_flag:
            ret, frame = self.videoCapture.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)

        if not self.get_points_flag:
            self.line_button.setText("Submit Area")
            self.get_points_flag = 1
            self.pntnum = 0
            self.pointarray = []
            self.linearray = []
            self.open_button.setEnabled(False)
            self.start_button.setEnabled(False)
        else:
            self.line_button.setText("Select Area")
            self.get_points_flag = 0
            print(str(self.linearray))
            # Enable start button
            self.open_button.setEnabled(True)
            self.start_button.setEnabled(True)

    def export(self):
        with open("./count.csv", 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['', 'Car', 'Bus', 'Truck', 'Minibus', 'Sum'])
            for i in range(self.cnt):
                for j in range(self.cnt):
                    if i == j: continue
                    sumcnt = self.carcnt[i][j] + self.buscnt[i][j] + self.truckcnt[i][j] + self.minibuscnt[i][i]
                    csvwriter.writerow([f'{i}->{j}', self.carcnt[i][j], self.buscnt[i][j], self.truckcnt[i][j], self.minibuscnt[i][j], sumcnt])

            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())