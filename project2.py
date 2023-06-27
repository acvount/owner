import os
import sys
import mss
import cv2 
import time
import json
from PIL import Image
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch

config_file_path = './conf.json'
default_title = 'ai-project-fps'

class Utils:
    conf = None

    def __getitem__(self, key):
        return self.conf[key]
    
    def __init__(self) -> None:
        self.conf = self.create_or_read_conf_file()

    def _get_default_conf(self):
        return {
            'use_gpu': True,
            'gpu_model': 'mps',
            'show_fps': True,
            'show_box': True,
            'show_label': True,
            'confidence_threshold': 0.5,
            'capture_width': 200,
            'capture_height': 200,
            'zoom_factor': 3,
        }

    def _check_conf_file_exist(self):
        return os.path.isfile(config_file_path)

    def create_or_read_conf_file(self):
        if self._check_conf_file_exist():
            with open(config_file_path, 'r') as file:
                return json.load(file)
        else:
            conf = self._get_default_conf()
            self._write_conf_file(conf)
            return conf

    def _write_conf_file(self, data):
        with open(config_file_path, 'w') as file:
            json.dump(data, file)

    def capture_center_image(self, width, height):
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

    def display_image(self, image, label):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # 将图像转换为Pixmap
        pixmap = QPixmap.fromImage(q_image)

        # 调整图像大小以适应标签
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)

        # 在标签中显示图像
        label.setPixmap(scaled_pixmap)

    def scale_image(self, image, scale_factor):
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = image.resize((new_width, new_height))
        return resized_image

class  MainWindow(QMainWindow):
    conf = None
    model = None

    def __init__(self):
        super().__init__()
        self.conf = Utils()
        # gui初始化
        self._gui_init()
        # 通用类初始化
        self._load_model()
        
        self.capture_thread = CaptureThread(self.conf, self.model, self.conf['gpu_model'] if self.conf['use_gpu'] else 'cpu')
        self.capture_thread.frame_captured.connect(self.process_frame)
        self.capture_thread.start()
    
    def _gui_init(self):
        self.setWindowTitle(default_title)
        self.setGeometry(100, 100, 400, 300)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # 创建QLabel组件用于显示图像
        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 10, 380, 280)

        if self.conf['show_fps']:
            self.frame_rate = 0
            self.frame_count = 0
            self.start_time = time.time()

    def _load_model(self):
        # 加载模型
        if self.conf['use_gpu']:
            self.device = self.conf['gpu_model']
        else:
            self.device = 'cpu'
        self.model = YOLO('./yolov8s.pt')

    def process_frame(self, image):
        # 图像处理完成后更新标签中的图像
        self.conf.display_image(image, self.image_label)

    def resizeEvent(self, event):
        # 调整image_label的大小以适应窗体大小
        label_width = self.width() - 20  # 10px的边距
        label_height = self.height() - 20  # 10px的边距
        self.image_label.setGeometry(10, 10, label_width, label_height)
        event.accept()
   
 
class CaptureThread(QThread):
    frame_captured = pyqtSignal(object)

    def __init__(self, conf, model, device):
        super().__init__()
        self.conf = conf
        self.model = model
        self.device = device
        
        if self.conf['show_fps']:
            self.frame_rate = 0
            self.frame_count = 0
            self.start_time = time.time()
    

    def run(self):
        while True:
            # 截取图像
            image = self.conf.capture_center_image(self.conf['capture_width'], self.conf['capture_height'])
            scaled_image = self.conf.scale_image(image, self.conf['zoom_factor'])
            results = self.model(scaled_image, conf=self.conf['confidence_threshold'], device=self.device)[0]
            boxes = results.boxes
            names = results.names
            image = results.orig_img
            if boxes is not None:
                for box in reversed(boxes):
                    if names[int(box.cls)] == 'person':
                        # 获取置信度
                        conf = int(box.conf.squeeze() * 100)
                        # 获取坐标
                        x1, y1, x2, y2 = (int(i) for i in box.xyxy.squeeze())
                        # 显示框
                        if self.conf['show_label']:
                            cv2.putText(image, f'{names[int(box.cls)]} {conf}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            self.frame_captured.emit(image)

            if self.conf['show_fps']:
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= 1:
                    self.frame_rate = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                    print(f'FPS: {self.frame_rate:.2f}')

            time.sleep(1 / self.conf['fps'])


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
