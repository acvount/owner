from ultralytics import YOLO
import cv2 
# Load a model
model = YOLO('./yolov8x.pt')  # load an official detection model

# Track with the model
results = model(f'bus.jpg')[0]
boxes = results.boxes
names = results.names
img = results.orig_img
if boxes is not None:
    for box in reversed(boxes):
        conf = int(box.conf.squeeze() * 100)
        x1,y1,x2,y2 = (int(i) for i in box.xyxy.squeeze())
        cv2.putText(img, f'{names[int(box.cls)]} {conf}%', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 4)
        print(f'{names[int(box.cls)]}  {x1}-{y1}-{x2}-{y2}')
        print('=========================================================')
    cv2.imshow('img', img)
    cv2.waitKey(0)