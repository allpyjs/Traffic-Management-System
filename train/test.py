from ultralytics import YOLO
import cv2
import numpy as np
# Load a model
model = YOLO("../model/best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images

image = cv2.imread('3.png')
image = cv2.resize(image, [640, 480])

# results = model(["1.jpg", "2.jpg"])  # return a list of Results objects
results = model(image)
# Process results list

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs

    print (len(boxes))

    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk