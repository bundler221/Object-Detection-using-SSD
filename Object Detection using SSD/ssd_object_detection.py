from imutils.video import FPS
from tkinter import *
import numpy as np
import imutils
import cv2
import math

use_gpu = True
live_video = False



confidence_level = 0.6
temp=int(0)
intu=int(0)
tec=""
text=""""""
fps = FPS().start()
ret = True
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor","gun","revolver"]

COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 1))

net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

if use_gpu:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
image = cv2.imread("testimage.png")

print("[INFO] accessing video stream...")
if live_video:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture('test.mp4')

while ret:
    ret, frame = vs.read()
    if ret:
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        image = cv2.imread("testimage.png")

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            image = cv2.imread("testimage.png")
            
            if confidence > confidence_level:
                idx = int(detections[0, 0, i, 1])
                intu = idx
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}1".format(CLASSES[idx])
                if tec == CLASSES[idx]:
                    temp=temp +1
                    label = "{}2".format(CLASSES[idx])
                temp=temp+1
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                tec=str(CLASSES[idx])
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)
                text = "No of "+tec+" = "+str(math.ceil(temp/2))
                     
        temp=int(0)
        frame = imutils.resize(frame,height=400)
        cv2.imshow('Live detection',frame)
        coordinates = (1,15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = (255,0,255)
        thickness = 1
        y=int(15)
        for i in text:
            if i=="\n":
                y=y+15
        cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_DUPLEX, 0.3, 0, 1)
        image = imutils.resize(image,height=200)
        cv2.imshow("Text", image)
        
        if cv2.waitKey(1)==27:
            break

        fps.update()
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#print("Total no of Objects :",temp)
print(type(temp))

