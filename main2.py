import sys
import mss
import cv2 
import time
from PIL import Image
from ultralytics import YOLO

model = YOLO('/Users/acfan/workspace/python/fps/project/owner/yolov8x.pt')

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


def display_image(image):
    # 将图像转换为OpenCV格式
    image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 显示图像
    cv2.imshow("Image", image_cv2)


# 创建计时器
frame_count = 0
start_time = time.time()

# 进入循环，每隔一段时间更新图像
while True:
    # 截取图像
    image = capture_center_image(480, 640)
    results = model(image,conf=0.7,device='mps')[0]
    boxes = results.boxes
    names = results.names
    image = results.orig_img
    if boxes is not None:
        for box in reversed(boxes):
            if names[int(box.cls)] == 'person' :
                conf = int(box.conf.squeeze() * 100)
                x1, y1, x2, y2 = (int(i) for i in box.xyxy.squeeze())
                cv2.putText(image, f'{names[int(box.cls)]} {conf}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                # print(f'{names[int(box.cls)]}  {x1}-{y1}-{x2}-{y2}')

    # 显示图像
    display_image(image)

    # 更新帧率计数器
    frame_count += 1
    elapsed_time = time.time() - start_time

    # 计算帧率
    if elapsed_time >= 1.0:
        frame_rate = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        print(f"帧率: {frame_rate:.2f}")

    # 退出循环按键
    if cv2.waitKey(1) == ord('q'):
        break

# 关闭窗口
cv2.destroyAllWindows()
