import sys
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QWidget, QFileDialog, QSlider, QLayout, QGraphicsView, QGraphicsScene, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from PyQt5.QtCore import QUrl, QDir, Qt, QSizeF, pyqtSignal
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap
from moviepy.editor import VideoFileClip
import os, cv2
import numpy as np
from base import Ui_Dialog
from ultralytics import YOLO
from sort import *
import cvzone

import math
class VideoPlayer(QDialog, Ui_Dialog):

    def __init__(self, *args, obj = None, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.image_size = [self.image.geometry().width(), self.image.geometry().height()]

        self.model = YOLO("../model/nbest.pt")
        self.classNames = ['truck', 'car', 'minibus', 'bus'
        ]
        self.initsignal()
        self.connectsignal()

        self.running_flag = 0
        self.pause_flag = 0
        self.get_points_flag = 0
        self.counter_thread_start_flag = 0

        self.pntnum = 0
        self.pointarray = [[]]
        self.linearray = [[]]

        self.video_paused = False
        
    def connectsignal(self):
        self.open_button.clicked.connect(self.open_file)
        self.start_button.clicked.connect(self.start_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.line_button.clicked.connect(self.process)

        self.image.mouseDoubleClickEvent = self.get_points

    def initsignal(self):
        self.changesignal2(False)
        self.changesignal3(False)
        self.changesignal4(False)


    def closeEvent(self, event):
        # Release the video capture when the window is closed
        self.release_video_capture()
        event.accept()

    def release_video_capture(self):
        # Release the video capture if it exists
        if hasattr(self, 'video_capture') and self.cap.isOpened():
            self.video_capture.release()

    def changesignal2(self, state):
        self.dir_1_2.setVisible(state)
        self.dir_2_1.setVisible(state)

        self.car_1_2.setVisible(state)
        self.car_2_1.setVisible(state)

        self.bus_1_2.setVisible(state)
        self.bus_2_1.setVisible(state)

        self.truck_1_2.setVisible(state)
        self.truck_2_1.setVisible(state)

        self.minibus_1_2.setVisible(state)
        self.minibus_2_1.setVisible(state)

        self.sum_1_2.setVisible(state)
        self.sum_2_1.setVisible(state)

    def changesignal3(self, state):
        self.changesignal2(state)
        self.dir_1_3.setVisible(state)
        self.dir_3_1.setVisible(state)
        self.dir_2_3.setVisible(state)
        self.dir_3_2.setVisible(state)

        self.car_1_3.setVisible(state)
        self.car_2_3.setVisible(state)
        self.car_3_1.setVisible(state)
        self.car_3_2.setVisible(state)

        self.bus_1_3.setVisible(state)
        self.bus_2_3.setVisible(state)
        self.bus_3_1.setVisible(state)
        self.bus_3_2.setVisible(state)

        self.truck_1_3.setVisible(state)
        self.truck_2_3.setVisible(state)
        self.truck_3_1.setVisible(state)
        self.truck_3_2.setVisible(state)

        self.minibus_1_3.setVisible(state)
        self.minibus_2_3.setVisible(state)
        self.minibus_3_1.setVisible(state)
        self.minibus_3_2.setVisible(state)

        self.sum_1_3.setVisible(state)
        self.sum_2_3.setVisible(state)
        self.sum_3_1.setVisible(state)
        self.sum_3_2.setVisible(state)

    def changesignal4(self, state):
        self.changesignal2(state)
        self.changesignal3(state)
        self.dir_1_4.setVisible(state)
        self.dir_4_1.setVisible(state)
        self.dir_4_2.setVisible(state)
        self.dir_2_4.setVisible(state)
        self.dir_3_4.setVisible(state)
        self.dir_4_3.setVisible(state)

        self.car_1_4.setVisible(state)
        self.car_2_4.setVisible(state)
        self.car_3_4.setVisible(state)
        self.car_4_1.setVisible(state)
        self.car_4_2.setVisible(state)
        self.car_4_3.setVisible(state)

        self.bus_1_4.setVisible(state)
        self.bus_2_4.setVisible(state)
        self.bus_3_4.setVisible(state)
        self.bus_4_1.setVisible(state)
        self.bus_4_2.setVisible(state)
        self.bus_4_3.setVisible(state)

        self.truck_1_4.setVisible(state)
        self.truck_2_4.setVisible(state)
        self.truck_3_4.setVisible(state)
        self.truck_4_1.setVisible(state)
        self.truck_4_2.setVisible(state)
        self.truck_4_3.setVisible(state)

        self.minibus_1_4.setVisible(state)
        self.minibus_2_4.setVisible(state)
        self.minibus_3_4.setVisible(state)
        self.minibus_4_1.setVisible(state)
        self.minibus_4_2.setVisible(state)
        self.minibus_4_3.setVisible(state)

        self.sum_1_4.setVisible(state)
        self.sum_2_4.setVisible(state)
        self.sum_3_4.setVisible(state)
        self.sum_4_1.setVisible(state)
        self.sum_4_2.setVisible(state)
        self.sum_4_3.setVisible(state)
        

    def open_file(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie", QDir.homePath())
        
        cap = cv2.VideoCapture(self.fileName)
        while cap.isOpened():
            ret, self.frame = cap.read()
            if ret:
                # self.exampleImage = self.frame
                self.show_image(self.frame)
                self.imgScale = np.array(self.frame.shape[:2]) / [self.image_size[1], self.image_size[0]]
                cap.release()
                break
            
    def show_image(self, img_np):
        img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, self.image_size)
        bytesPerLine = 3 * self.image_size[0]
        frame = QImage(img_np.data, self.image_size[0], self.image_size[1], bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        self.image.setPixmap(pix)
        self.image.repaint()

    def start_video(self):
        cnt = self.pntnum / 2
        if cnt == 2:
            self.changesignal2(True)
        elif cnt == 3:
            self.changesignal3(True)
        elif cnt == 4:
            self.changesignal4(True)
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        limitsUp = [250, 400, 575, 400]
        limitsDown = [700, 450, 1125, 450]
        total_countsUp = {"car":[], "bus":[], "truck":[]}
        total_countsDown = {"car":[], "bus":[], "truck":[]}
        print(self.linearray)
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
                for i in range(1, len(self.linearray)):
                    cv2.line(img, (self.linearray[i][0], self.linearray[i][1]), (self.linearray[i][2], self.linearray[i][3]), color=(0, 0, 0), thickness= 1)
                results = self.model(img, stream=True)
                detections = np.empty((0, 5))
                for r in results:
                    boxes = r.boxes
                    print ("length of boxes: ", len(boxes))
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
                        currentclass = self.classNames[cls]
                        if currentclass == 'car' or currentclass == 'bus'\
                            or currentclass == 'truck' or currentclass == 'minibus'\
                                and conf > 0.4:
                            cvzone.cornerRect(img, (x1, y1, w, h) , l=15)
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
                    cx, cy = x1+w//2 , y1+h//2
                    cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

                self.show_image(img)
                cv2.waitKey(1)
                cv2.destroyAllWindows()


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

        if self.pntnum % 2 == 0:
            cv2.line(self.frame, (self.pointarray[-1][0], self.pointarray[-1][1]), (realx, realy), (255, 0, 0), 2)
            self.show_image(self.frame)
            self.linearray.append([self.pointarray[-1][0], self.pointarray[-1][1], realx, realy])
            print(self.linearray[-1])
        self.pointarray.append([realx,realy])

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
            self.pointarray = [[]]
            self.linearray = [[]]
            self.open_button.setEnabled(False)
            self.start_button.setEnabled(False)
        else:
            self.line_button.setText("Select Area")
            self.get_points_flag = 0
            #enable start button
            self.open_button.setEnabled(True)
            self.start_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
