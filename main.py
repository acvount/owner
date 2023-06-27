import sys
import mss
import cv2 
import time
from PIL import Image
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch

device = torch.device("mps")
model = YOLO('/Users/acfan/workspace/python/fps/project/owner/yolov8n.pt')
model.to(device)
# 加载模型
# model = YOLO('./yolov8x.pt')  # load an official detection model


def capture_center_image(width, height):
    with mss.mss() as sct:
        # 获取屏幕信息
        monitor = sct.monitors[1]  # 如果有多个屏幕，选择你想要截取的屏幕
        screen_width = monitor["width"]
        screen_height = monitor["height"]

        # 计算中心范围的坐标
        capture_x = (screen_width - width) // 2
        capture_y = (screen_height - height) // 2

        # 设置截取的区域
        monitor["left"] = capture_x
        monitor["top"] = capture_y
        monitor["width"] = width
        monitor["height"] = height

        # 截取中心范围的图像
        screen_shot = sct.grab(monitor)

        # 将截图转换为PIL图像
        image = Image.frombytes("RGB", screen_shot.size, screen_shot.rgb)

    return image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("demo")
        self.setGeometry(100, 100, 400, 300)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # 创建QLabel组件用于显示图像
        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 10, 380, 280)

        self.frame_rate = 0
        self.frame_count = 0
        self.start_time = time.time()

    def update_image(self):
        # 截取图像
        image = capture_center_image(600, 600)
        results = model(image,conf=0.5)[0]
        boxes = results.boxes
        names = results.names
        image = results.orig_img
        if boxes is not None:
            for box in reversed(boxes):
                conf = int(box.conf.squeeze() * 100)
                x1,y1,x2,y2 = (int(i) for i in box.xyxy.squeeze())
                cv2.putText(image, f'{names[int(box.cls)]} {conf}%', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 4)
                # print(f'{names[int(box.cls)]}  {x1}-{y1}-{x2}-{y2}')
        
        # 转换图像格式
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, width * channels, QImage.Format_RGB888).copy()

        # 创建QPixmap对象并设置图像
        pixmap = QPixmap.fromImage(qimage)

        # 将图像显示在QLabel组件中
        self.image_label.setPixmap(pixmap.scaled(380, 280, Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.SmoothTransformation))

        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time >= 1.0:
            self.frame_rate = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        self.setWindowTitle(f"监控中 - 帧率: {self.frame_rate:.2f}")


# 创建应用程序对象
app = QApplication(sys.argv)

# 创建主窗体
window = MainWindow()
window.show()

# 进入循环，每隔一段时间更新图像
while True:
    window.update_image()
    app.processEvents()  # 处理GUI事件

# 运行应用程序的主循环
sys.exit(app.exec())