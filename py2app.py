from setuptools import setup

APP = ['your_script.py']
DATA_FILES = [('.', ['yolov8s.pt'])]  # 添加模型文件或其他资源文件

OPTIONS = {
    'argv_emulation': True,
    'plist': {
        'CFBundleName': 'acfan',
        'CFBundleDisplayName': 'acfan',
        'CFBundleGetInfoString': 'acfan',
        'CFBundleIdentifier': 'com.acvount.ai-bot',
        'CFBundleVersion': '0.1',
        'CFBundleShortVersionString': '0.1',
        'LSMinimumSystemVersion': '10.14',
    },
    'packages': ['torch','YOLOv8','PIL','json','time','cv2','mss','sys','os'],  # 添加所需的依赖包
    'iconfile': '/Users/acfan/workspace/python/fps/project/owner/IMG_0994.jpeg',  # 添加应用程序图标文件
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
