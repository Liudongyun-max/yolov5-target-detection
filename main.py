import os
import sys
from pathlib import Path
import cv2
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtMultimedia import QCamera, QCameraInfo, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder

from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
from utils.augmentations import letterbox
from utils.plots import plot_one_box

from models.common import DetectMultiBackend


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class VideoThread(QThread):
    """视频处理线程，用于处理摄像头或视频的每一帧"""
    update_frame = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(VideoThread, self).__init__(parent)
        self.mutex = QMutex()
        self.cap = None
        self.is_running = False
        self.model = None
        self.device = None
        self.imgsz = 640
        self.half = False
        self.names = []
        self.colors = []
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.process_frequency = 1  # 每隔多少帧处理一次，对视频设为1以确保每帧都处理
        self.frame_count = 0
        self.video_path = None
        self.is_video_file = False
        
    def setup(self, model, device, imgsz, half, names, colors, conf_thres, iou_thres):
        """设置模型和参数"""
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.names = names
        self.colors = colors
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
    def set_camera(self, camera_id=0):
        """设置摄像头或视频文件
        Args:
            camera_id: 如果是整数则表示摄像头ID，如果是字符串则表示视频文件路径
        """
        try:
            self.mutex.lock()
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # 检测是否为视频文件
            self.is_video_file = isinstance(camera_id, str) and os.path.isfile(camera_id)
            
            if self.is_video_file:
                self.video_path = camera_id
                print(f"正在打开视频文件: {self.video_path}")
                self.cap = cv2.VideoCapture(self.video_path)
                self.process_frequency = 1  # 视频文件处理每一帧
            else:
                print(f"正在打开摄像头 ID: {camera_id}")
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                self.process_frequency = 3  # 摄像头处理可以跳帧以提高性能
            
            if not self.cap.isOpened():
                error_msg = "无法打开摄像头设备" if not self.is_video_file else f"无法打开视频文件: {self.video_path}"
                self.error_signal.emit(error_msg)
                self.mutex.unlock()
                return False
                
            # 获取视频属性
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            device_type = "摄像头" if not self.is_video_file else "视频文件"
            print(f"{device_type}已打开：分辨率 {width}x{height}, FPS: {fps}")
            self.mutex.unlock()
            return True
        except Exception as e:
            error_msg = f"设置{'视频' if self.is_video_file else '摄像头'}出错: {str(e)}"
            self.error_signal.emit(error_msg)
            
            if self.mutex.tryLock():
                self.mutex.unlock()
            return False
            
    def run(self):
        """线程运行函数"""
        self.is_running = True
        self.frame_count = 0
        
        while self.is_running:
            try:
                self.mutex.lock()
                if self.cap is None or not self.cap.isOpened():
                    self.mutex.unlock()
                    # 视频文件结束
                    if self.is_video_file:
                        print("视频文件处理完毕")
                        # 发送信号通知主线程
                        self.error_signal.emit("VIDEO_END")
                    else:
                        print("摄像头已断开")
                        self.error_signal.emit("摄像头连接丢失")
                    break
                    
                ret, frame = self.cap.read()
                self.mutex.unlock()
                
                if not ret or frame is None:
                    # 视频结束或摄像头错误
                    if self.is_video_file:
                        print("视频文件读取完毕")
                        self.error_signal.emit("VIDEO_END")
                    else:
                        self.error_signal.emit("无法获取摄像头画面")
                    break
                
                self.frame_count += 1
                
                # 根据设置决定是否处理当前帧
                if self.frame_count % self.process_frequency == 0:
                    # 进行目标检测
                    try:
                        processed_frame = self.process_frame(frame)
                        if processed_frame is not None:
                            self.update_frame.emit(processed_frame)
                        else:
                            # 如果处理失败，仍然显示原始帧
                            self.update_frame.emit(frame)
                    except Exception as e:
                        print(f"处理帧时出错: {str(e)}")
                        # 出错时仍然显示原始帧
                        self.update_frame.emit(frame)
                else:
                    # 直接发送原始帧
                    self.update_frame.emit(frame)
                    
                # 摄像头模式下适当休眠，避免过度占用CPU
                # 视频文件模式下可以不休眠，以最大速度处理
                if not self.is_video_file:
                    self.msleep(10)
                
            except Exception as e:
                print(f"视频处理线程出错: {str(e)}")
                self.error_signal.emit(f"处理视频帧时出错: {str(e)}")
                if self.mutex.tryLock():
                    self.mutex.unlock()
                break
                
        print("视频处理线程已停止")
        
    def process_frame(self, frame):
        """处理单帧图像"""
        try:
            if self.model is None:
                return frame  # 如果模型未加载，直接返回原始帧
                
            with torch.no_grad():
                # 创建原始图像的副本用于显示
                showimg = frame.copy()
                
                # 预处理图像
                img = letterbox(frame, new_shape=self.imgsz)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # 进行推理
                pred = self.model(img)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                
                # 处理检测结果
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(
                                xyxy, showimg, label=label,
                                color=self.colors[int(cls)], line_thickness=2)
                
                return showimg
        except RuntimeError as e:
            # 处理CUDA错误
            if "CUDA out of memory" in str(e):
                self.error_signal.emit("GPU内存不足，请尝试降低图像分辨率或使用CPU模式")
            else:
                self.error_signal.emit(f"目标检测处理出错: {str(e)}")
            return None
        except Exception as e:
            self.error_signal.emit(f"目标检测处理出错: {str(e)}")
            return None
    
    def stop(self):
        """停止线程"""
        self.is_running = False
        self.wait(1000)  # 等待最多1秒线程结束
        
        try:
            self.mutex.lock()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.mutex.unlock()
        except Exception as e:
            print(f"释放视频资源时出错: {str(e)}")
            if self.mutex.tryLock():
                self.mutex.unlock()


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = None
        self.out = None
        self.is_camera_open = False
        self.frame_count = 0
        self.max_frames = 1000
        self.last_frame_time = 0
        self.frame_interval = 30
        
        # 创建视频处理线程
        self.video_thread = VideoThread()
        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.error_signal.connect(self.show_error)
        
        # GPU设置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            print("使用CPU进行检测")
            self.device = torch.device("cpu")
        
        self.half = torch.cuda.is_available()

        name = 'exp'
        save_file = ROOT / 'result'
        self.save_file = increment_path(Path(save_file) / name, exist_ok=False, mkdir=True)

        weights = 'weights/best.pt'   # 模型加载路径
        imgsz = 640  # 预测图尺寸大小
        self.conf_thres = 0.4  # NMS置信度
        self.iou_thres = 0.5  # IOU阈值

        # 载入模型
        try:
            self.model = DetectMultiBackend(weights, device=self.device)
            stride = self.model.stride
            self.imgsz = check_img_size(imgsz, s=stride)
            
            if self.half:
                self.model.half()  # 转换为FP16
                # 预热模型
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
            
            # 从模型中获取各类别名称
            self.names = self.model.names
            # 给每一个类别初始化颜色
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            
            # 设置线程的模型和参数
            self.video_thread.setup(
                self.model, self.device, self.imgsz, self.half,
                self.names, self.colors, self.conf_thres, self.iou_thres
            )
            
            print("模型加载成功，使用设备:", self.device)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "错误", 
                f"模型加载失败: {str(e)}\n请检查模型文件是否存在且格式正确",
                QtWidgets.QMessageBox.Ok
            )
            sys.exit(1)

        # 摄像头对象
        self.camera = None
        self.viewfinder = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)  # 增加窗口大小
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("#centralwidget{border-image:url(./UI/paddy.jpg)}")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        # self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")


        # 打开单图片按钮
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_img.setMaximumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_img.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)  # 调整字体大小
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(8)  # 增加按钮之间的间距
        self.pushButton_img.setToolTip('<b>请选择一张图片进行检测</b>')  # 创建提示框

        # 打开多图片按钮
        self.pushButton_imgs = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_imgs.sizePolicy().hasHeightForWidth())
        self.pushButton_imgs.setSizePolicy(sizePolicy)
        self.pushButton_imgs.setMinimumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_imgs.setMaximumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_imgs.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)  # 调整字体大小
        self.pushButton_imgs.setFont(font)
        self.pushButton_imgs.setObjectName("pushButton_imgs")
        self.verticalLayout.addWidget(self.pushButton_imgs, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(8)  # 增加按钮之间的间距
        self.pushButton_imgs.setToolTip('<b>请选择一张或多张图片进行检测</b>')  # 创建提示框

        # 打开图片文件按钮
        self.pushButton_imgfile = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_imgfile.sizePolicy().hasHeightForWidth())
        self.pushButton_imgfile.setSizePolicy(sizePolicy)
        self.pushButton_imgfile.setMinimumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_imgfile.setMaximumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_imgfile.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)  # 调整字体大小
        self.pushButton_imgfile.setFont(font)
        self.pushButton_imgfile.setObjectName("pushButton_imgfile")
        self.verticalLayout.addWidget(self.pushButton_imgfile, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(8)  # 增加按钮之间的间距
        self.pushButton_imgfile.setToolTip('<b>请选择包含所有检测图片的文件夹</b>')  # 创建提示框

        # 打开摄像头按钮
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_camera.setMaximumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_camera.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(8)
        self.pushButton_camera.setToolTip('<b>请确保摄像头设备正常</b>')  # 创建提示框

        # 打开视频按钮
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_video.setMaximumSize(QtCore.QSize(180, 45))  # 增加按钮宽度
        self.pushButton_video.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(50)
        self.pushButton_video.setToolTip('<b>请选择一个视频进行检测</b>')  # 创建提示框

        # 显示导出文件夹按钮
        self.pushButton_showdir = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_showdir.sizePolicy().hasHeightForWidth())
        self.pushButton_showdir.setSizePolicy(sizePolicy)
        self.pushButton_showdir.setMinimumSize(QtCore.QSize(180, 50))  # 增加按钮宽度
        self.pushButton_showdir.setMaximumSize(QtCore.QSize(180, 50))  # 增加按钮宽度
        self.pushButton_showdir.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        self.pushButton_showdir.setFont(font)
        self.pushButton_showdir.setObjectName("pushButton_showdir")
        self.verticalLayout.addWidget(self.pushButton_showdir, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_showdir.setToolTip('<b>检测后的文件将在此保存</b>')  # 创建提示框

        # 右侧图片/视频填充区域
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setMinimumSize(QtCore.QSize(800, 600))  # 设置最小显示尺寸
        self.label.setAlignment(QtCore.Qt.AlignCenter)  # 居中对齐
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 5)  # 增加右侧显示区域的比例
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.label.setStyleSheet("border: 1px solid white; border-radius: 5px; background-color: rgba(0, 0, 0, 0.1);")  # 添加半透明背景

        # 底部美化导航条
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLOv5目标检测平台"))
        self.pushButton_img.setText(_translate("MainWindow", "单图片检测"))
        self.pushButton_imgs.setText(_translate("MainWindow", "多图片检测"))
        self.pushButton_imgfile.setText(_translate("MainWindow", "文件夹图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.pushButton_showdir.setText(_translate("MainWindow", "显示输出文件夹"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.timer_video.timeout.connect(self.show_video_frame)
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.pushButton_imgs.clicked.connect(self.button_images_open)
        self.pushButton_imgfile.clicked.connect(self.button_imagefile_open)

    def init_logo(self):
        pix = QtGui.QPixmap('')   # 绘制初始化图片
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    # 退出提示
    def closeEvent(self, event):
        """重写关闭事件，确保资源正确释放"""
        try:
            self.stop_video_processing()
            self.stop_camera()
        except:
            pass
        reply = QtWidgets.QMessageBox.question(
            self, '提示',
            "确定要退出吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def button_image_open(self):
        print('打开图片')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            self.empty_information()
            print('empty!')
            return
        
        try:
            img = cv2.imread(img_name)
            if img is None:
                raise ValueError("无法读取图片")
                
            print(img_name)
            showimg = img
            with torch.no_grad():
                # 确保输入数据在正确的设备上
                img = letterbox(img, new_shape=self.imgsz)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # 使用GPU进行推理
                pred = self.model(img)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            cv2.imwrite(str(Path(self.save_file / 'prediction.jpg')), showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            
            # 计算缩放比例，确保图片完整显示
            label_size = self.label.size()
            img_height, img_width = self.result.shape[:2]
            scale = min(label_size.width() / img_width, label_size.height() / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            self.result = cv2.resize(self.result, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            print('单图片检测完成')
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "错误",
                f"处理图片时出错: {str(e)}",
                QtWidgets.QMessageBox.Ok
            )

    def button_images_open(self):
        print('打开图片')

        img_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if len(img_names) == 0:
            self.empty_information()
            print('empty!')
            return
        index = 0
        for img_name in img_names:
            name_list = []
            img = cv2.imread(img_name)
            print(img_name)
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            # print(label.split()[0])  # 打印各目标名称
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            cv2.imwrite(str(Path(self.save_file / 'prediction_imgs{}.jpg'.format(index))), showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            index += 1
        print('多图片检测完成')

    def button_imagefile_open(self):
        print('打开图片文件夹')
        try:
            # 记录开始时间
            start_time = QtCore.QTime.currentTime()
            
            file_name = QtWidgets.QFileDialog.getExistingDirectory(
                self, "打开图片文件夹", "")
            if not file_name:
                self.empty_information()
                print('empty!')
                return
                
            print(file_name)
            img_names = [f for f in os.listdir(file_name) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if len(img_names) == 0:
                self.empty_information()
                print('文件夹中没有图片文件!')
                return
                
            # 创建进度对话框
            progress = QtWidgets.QProgressDialog("正在处理图片...", "取消", 0, len(img_names), self)
            progress.setWindowTitle("进度")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            
            for index, img_name in enumerate(img_names):
                if progress.wasCanceled():
                    break
                    
                try:
                    # 更新进度
                    progress.setValue(index)
                    progress.setLabelText(f"正在处理: {img_name}")
                    QtWidgets.QApplication.processEvents()  # 保持界面响应
                    
                    name_list = []
                    img_path = os.path.join(file_name, img_name)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"无法读取图片: {img_name}")
                        continue
                        
                    print(f"处理图片: {img_name}")
                    showimg = img.copy()
                    
                    with torch.no_grad():
                        # 预处理图像
                        img = letterbox(img, new_shape=self.imgsz)[0]
                        img = img[:, :, ::-1].transpose(2, 0, 1)
                        img = np.ascontiguousarray(img)
                        img = torch.from_numpy(img).to(self.device)
                        img = img.half() if self.half else img.float()
                        img /= 255.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)
                            
                        # 使用GPU进行推理
                        pred = self.model(img)[0]
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                        
                        # 处理检测结果
                        for i, det in enumerate(pred):
                            if det is not None and len(det):
                                det[:, :4] = scale_boxes(
                                    img.shape[2:], det[:, :4], showimg.shape).round()
                                
                                for *xyxy, conf, cls in reversed(det):
                                    label = '%s %.2f' % (self.names[int(cls)], conf)
                                    name_list.append(self.names[int(cls)])
                                    plot_one_box(xyxy, showimg, label=label,
                                                 color=self.colors[int(cls)], line_thickness=2)
                    
                    # 保存检测结果
                    cv2.imwrite(str(Path(self.save_file / f'prediction_file{index}.jpg')), showimg)
                    
                    # 实时显示检测结果
                    self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
                    
                    # 计算缩放比例，确保图片完整显示
                    label_size = self.label.size()
                    img_height, img_width = self.result.shape[:2]
                    scale = min(label_size.width() / img_width, label_size.height() / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    
                    self.result = cv2.resize(self.result, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],QtGui.QImage.Format_RGB32)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    
                    # 短暂延迟，让用户能看到每张图片的检测结果
                    QtCore.QThread.msleep(100)
                    QtWidgets.QApplication.processEvents()
                    
                except Exception as e:
                    print(f"处理图片 {img_name} 时出错: {str(e)}")
                    continue
                    
            progress.setValue(len(img_names))
            
            # 计算并输出总用时
            end_time = QtCore.QTime.currentTime()
            elapsed_time = start_time.msecsTo(end_time)
            hours = elapsed_time // 3600000
            minutes = (elapsed_time % 3600000) // 60000
            seconds = (elapsed_time % 60000) // 1000
            milliseconds = elapsed_time % 1000
            
            print(f'文件夹图片检测完成')
            print(f'总用时: {hours}小时 {minutes}分钟 {seconds}秒 {milliseconds}毫秒')
            print(f'平均每张图片用时: {elapsed_time/len(img_names):.2f}毫秒')
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "错误",
                f"处理文件夹时出错: {str(e)}",
                QtWidgets.QMessageBox.Ok
            )

    def button_video_open(self):
        """打开视频文件并进行实时检测"""
        try:
            # 如果有正在运行的视频或摄像头，先停止
            self.stop_video_processing()
            
            # 获取视频文件路径
            video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "打开视频", "", "视频文件 (*.mp4 *.avi);;所有文件 (*.*)"
            )
            
            if not video_name:
                self.empty_information()
                print('未选择视频文件')
                return
            
            # 创建视频捕获对象
            self.cap = cv2.VideoCapture(video_name)
            
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(
                    self, "警告", "无法打开视频文件，请检查文件是否损坏",
                    QtWidgets.QMessageBox.Ok
                )
                self.cap = None
                return
            
            # 获取视频属性
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: 分辨率={width}x{height}, FPS={fps}, 总帧数={total_frames}")
            
            # 创建输出视频
            os.makedirs(self.save_file, exist_ok=True)
            output_path = str(Path(self.save_file) / 'video_detection.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not self.out.isOpened():
                print(f"警告: 无法创建输出视频文件 {output_path}")
            
            # 设置视频处理线程
            if self.video_thread.set_camera(video_name):
                # 在线程中处理视频
                self.video_thread.start()
                
                # 禁用其他按钮
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_imgs.setDisabled(True)
                self.pushButton_imgfile.setDisabled(True)
                self.pushButton_camera.setDisabled(True)
                
                print(f"视频 {video_name} 开始处理")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "警告", "无法启动视频处理线程",
                    QtWidgets.QMessageBox.Ok
                )
                
                # 清理资源
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                if self.out is not None:
                    self.out.release()
                    self.out = None
        
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "错误", f"视频处理出错: {str(e)}",
                QtWidgets.QMessageBox.Ok
            )
            print(f"视频处理异常: {str(e)}")
            self.stop_video_processing()

    def show_video_frame(self):
        """从视频文件读取并显示下一帧（已弃用，使用VideoThread替代）"""
        try:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # 处理帧并显示
                    processed = self.process_frame(frame)
                    self.update_frame(processed if processed is not None else frame)
                else:
                    # 视频结束
                    self.stop_video_processing()
                    QtWidgets.QMessageBox.information(
                        self, "提示", "视频播放完成",
                        QtWidgets.QMessageBox.Ok
                    )
            else:
                self.stop_video_processing()
        except Exception as e:
            print(f"视频帧处理出错: {str(e)}")
            self.stop_video_processing()

    def process_frame(self, frame):
        """处理单个视频帧进行目标检测"""
        try:
            # 原始图像的副本用于显示
            display_img = frame.copy()
            
            # 目标检测
            with torch.no_grad():
                # 预处理图像
                input_img = letterbox(frame, new_shape=self.imgsz)[0]
                input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB并调整维度顺序
                input_img = np.ascontiguousarray(input_img)
                input_img = torch.from_numpy(input_img).to(self.device)
                input_img = input_img.half() if self.half else input_img.float()
                input_img /= 255.0
                if input_img.ndimension() == 3:
                    input_img = input_img.unsqueeze(0)
                
                # 推理
                pred = self.model(input_img)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                
                # 在原图上绘制检测结果
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_boxes(input_img.shape[2:], det[:, :4], display_img.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, display_img, label=label, color=self.colors[int(cls)], line_thickness=2)
            
            return display_img
        except Exception as e:
            print(f"处理视频帧时出错: {str(e)}")
            return None

    def stop_video_processing(self):
        """安全停止视频处理并释放资源"""
        try:
            # 停止视频线程
            if self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait()  # 等待线程结束
            
            # 停止定时器
            if self.timer_video.isActive():
                self.timer_video.stop()
            
            # 释放视频资源
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            if self.out is not None:
                self.out.release()
                self.out = None
            
            # 恢复按钮状态
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_imgs.setDisabled(False)
            self.pushButton_imgfile.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            
            # 恢复初始界面
            self.init_logo()
            
            print("视频处理已停止，资源已释放")
        except Exception as e:
            print(f"停止视频处理时出错: {str(e)}")

    def button_camera_open(self):
        """使用PyQt5原生摄像头功能并添加目标检测"""
        try:
            if not self.is_camera_open:
                print("正在打开摄像头...")
                
                # 禁用其他按钮
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_imgs.setDisabled(True)
                self.pushButton_imgfile.setDisabled(True)
                
                # 检查可用摄像头
                available_cameras = QCameraInfo.availableCameras()
                if not available_cameras:
                    QtWidgets.QMessageBox.warning(
                        self, "警告", "没有找到可用的摄像头设备", 
                        QtWidgets.QMessageBox.Ok)
                    self.stop_camera()
                    return
                
                print(f"找到 {len(available_cameras)} 个摄像头设备")
                for cam in available_cameras:
                    print(f"  - {cam.description()}")
                
                # 创建摄像头对象
                self.camera = QCamera(available_cameras[0])
                
                # 创建取景器
                if self.viewfinder is None:
                    self.viewfinder = QCameraViewfinder()
                    self.viewfinder.show()
                    
                    # 将取景器添加到label位置
                    self.label.setVisible(False)  # 隐藏原有label
                    self.horizontalLayout.removeWidget(self.label)
                    self.horizontalLayout.addWidget(self.viewfinder)
                
                # 设置取景器
                self.camera.setViewfinder(self.viewfinder)
                
                # 创建图像捕获对象
                self.camera_capture = QCameraImageCapture(self.camera)
                self.camera_capture.setCaptureDestination(QCameraImageCapture.CaptureToBuffer)
                self.camera_capture.imageCaptured.connect(self.process_captured_image)
                
                # 启动摄像头
                self.camera.start()
                
                # 设置状态
                self.is_camera_open = True
                self.detection_active = True
                self.capture_count = 0
                
                # 启动检测定时器 - 每200毫秒捕获一帧进行检测
                self.detection_timer = QtCore.QTimer()
                self.detection_timer.timeout.connect(self.capture_frame)
                self.detection_timer.start(200)  # 5fps检测频率，降低系统负担
                
                # 更新按钮文本
                self.pushButton_camera.setText("关闭摄像头")
                print("摄像头已打开，目标检测已启动")
            else:
                self.stop_camera()
                
        except Exception as e:
            print(f"打开摄像头时出错: {str(e)}")
            QtWidgets.QMessageBox.warning(
                self, "错误", f"摄像头操作出错: {str(e)}", 
                QtWidgets.QMessageBox.Ok)
            self.stop_camera()
    
    def capture_frame(self):
        """定时器触发函数，捕获当前帧进行检测"""
        if self.is_camera_open and self.camera and self.camera.status() == QCamera.ActiveStatus:
            self.capture_count += 1
            self.camera_capture.capture()
    
    def process_captured_image(self, id, image):
        """处理捕获的图像并进行目标检测"""
        try:
            # 将QImage转换为OpenCV格式
            img = self.qimage_to_opencv(image)
            if img is None:
                return
            
            # 原始图像的副本用于显示
            display_img = img.copy()
            
            # 目标检测
            with torch.no_grad():
                # 预处理图像
                input_img = letterbox(img, new_shape=self.imgsz)[0]
                input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB并调整维度顺序
                input_img = np.ascontiguousarray(input_img)
                input_img = torch.from_numpy(input_img).to(self.device)
                input_img = input_img.half() if self.half else input_img.float()
                input_img /= 255.0
                if input_img.ndimension() == 3:
                    input_img = input_img.unsqueeze(0)
                
                # 推理
                pred = self.model(input_img)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                
                # 在原图上绘制检测结果
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_boxes(input_img.shape[2:], det[:, :4], display_img.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, display_img, label=label, color=self.colors[int(cls)], line_thickness=2)
            
            # 将处理后的图像显示在界面上
            self.display_detection_result(display_img)
            
        except Exception as e:
            print(f"处理捕获图像时出错: {str(e)}")
    
    def qimage_to_opencv(self, qimage):
        """将QImage转换为OpenCV格式"""
        try:
            qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB888)
            width = qimage.width()
            height = qimage.height()
            
            # 创建NumPy数组
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            img = np.array(ptr).reshape(height, width, 3)
            
            # 转换为BGR格式（OpenCV默认格式）
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print(f"QImage转换为OpenCV格式时出错: {str(e)}")
            return None
    
    def display_detection_result(self, img):
        """在独立窗口中显示检测结果"""
        try:
            # 创建检测结果的QImage
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt_img = QtGui.QImage(rgb_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # 创建一个新窗口显示检测结果
            if not hasattr(self, 'detection_label'):
                self.detection_label = QtWidgets.QLabel()
                self.detection_label.setWindowTitle("目标检测结果")
                self.detection_label.resize(width, height)
                self.detection_label.show()
            
            # 更新图像
            self.detection_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))
            
        except Exception as e:
            print(f"显示检测结果时出错: {str(e)}")

    def update_frame(self, frame):
        """更新UI中显示的帧"""
        try:
            if frame is None:
                print("警告: 收到空帧")
                return
                
            if self.out is not None and self.out.isOpened():
                # 保存处理后的帧到输出视频
                try:
                    self.out.write(frame)
                except Exception as e:
                    print(f"写入输出视频时出错: {str(e)}")
            
            # 计算缩放比例，确保图片完整显示
            label_size = self.label.size()
            img_height, img_width = frame.shape[:2]
            scale = min(label_size.width() / img_width, label_size.height() / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 缩放图像
            try:
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"调整图像大小时出错: {str(e)}")
                return
                
            # 转换颜色格式
            try:
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"转换颜色格式时出错: {str(e)}")
                return
                
            # 显示图像
            try:
                qt_image = QtGui.QImage(
                    rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                    rgb_frame.shape[1] * 3, QtGui.QImage.Format_RGB888
                )
                self.label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            except Exception as e:
                print(f"创建QImage或设置Pixmap时出错: {str(e)}")
            
        except Exception as e:
            print(f"更新画面出错: {str(e)}")

    def show_error(self, error_message):
        """显示错误消息"""
        if error_message == "VIDEO_END":
            # 视频播放结束的特殊信号
            self.stop_video_processing()
            QtWidgets.QMessageBox.information(
                self, "提示", "视频处理完成", QtWidgets.QMessageBox.Ok
            )
        else:
            # 常规错误信息
            QtWidgets.QMessageBox.warning(
                self, "错误", error_message, QtWidgets.QMessageBox.Ok
            )

    def stop_camera(self):
        """安全关闭摄像头"""
        print("正在关闭摄像头...")
        try:
            # 停止检测定时器
            if hasattr(self, 'detection_timer') and self.detection_timer.isActive():
                self.detection_timer.stop()
            
            # 停止摄像头
            if self.camera is not None:
                self.camera.stop()
                self.camera = None
            
            # 关闭检测结果窗口
            if hasattr(self, 'detection_label'):
                self.detection_label.close()
                del self.detection_label
            
            # 恢复UI
            if self.viewfinder is not None:
                self.viewfinder.setVisible(False)
                self.horizontalLayout.removeWidget(self.viewfinder)
                self.horizontalLayout.addWidget(self.label)
                self.label.setVisible(True)
                self.init_logo()
            
            # 恢复按钮状态
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_imgs.setDisabled(False)
            self.pushButton_imgfile.setDisabled(False)
            self.pushButton_camera.setText("摄像头检测")
            
            # 重置状态
            self.is_camera_open = False
            
            print("摄像头已关闭")
        except Exception as e:
            print(f"关闭摄像头时出错: {str(e)}")

    def button_show_dir(self):
        path = self.save_file
        os.system(f"start explorer {path}")

    def empty_information(self):
        QtWidgets.QMessageBox.information(self, '提示', '未选择文件或选择文件为空!', QtWidgets.QMessageBox.Cancel)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui = Ui_MainWindow()
    # 设置窗口透明度
    ui.setWindowOpacity(1)
    # 去除顶部边框
    # ui.setWindowFlags(Qt.FramelessWindowHint)
    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon.ico"), QIcon.Normal, QIcon.Off)
    # 设置应用图标
    ui.setWindowIcon(icon)
    ui.show()
    sys.exit(app.exec_())