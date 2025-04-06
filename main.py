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

from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
from utils.augmentations import letterbox
from utils.plots import plot_one_box

from models.common import DetectMultiBackend


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = None  # 初始化为None
        self.out = None
        self.is_camera_open = False
        self.frame_count = 0
        self.max_frames = 1000  # 设置最大帧数，防止内存泄漏
        
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
            
            print("模型加载成功，使用设备:", self.device)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "错误", 
                f"模型加载失败: {str(e)}\n请检查模型文件是否存在且格式正确",
                QtWidgets.QMessageBox.Ok
            )
            sys.exit(1)

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
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_imgs.clicked.connect(self.button_images_open)
        self.pushButton_imgfile.clicked.connect(self.button_imagefile_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.pushButton_showdir.clicked.connect(self.button_show_dir)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('')   # 绘制初始化图片
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    # 退出提示
    def closeEvent(self, event):
        """重写关闭事件，确保资源正确释放"""
        try:
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
                    self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                             QtGui.QImage.Format_RGB32)
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
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")

        if not video_name:
            self.empty_information()
            print('empty!')
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter(str(Path(self.save_file / 'vedio_prediction.avi')), cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    def button_camera_open(self):
        try:
            if not self.is_camera_open:
                # 关闭之前的摄像头（如果存在）
                self.stop_camera()
                
                # 创建新的VideoCapture对象
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 使用DirectShow后端
                
                if not self.cap.isOpened():
                    QtWidgets.QMessageBox.warning(
                        self, u"警告", u"无法打开摄像头，请检查设备连接", 
                        buttons=QtWidgets.QMessageBox.Ok, 
                        defaultButton=QtWidgets.QMessageBox.Ok)
                    return
                
                # 设置摄像头参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                # 创建视频写入对象
                if self.out is not None:
                    self.out.release()
                self.out = cv2.VideoWriter(
                    str(Path(self.save_file / 'camera_prediction.avi')),
                    cv2.VideoWriter_fourcc(*'MJPG'),
                    20,
                    (640, 480)
                )
                
                self.frame_count = 0  # 重置帧计数器
                self.timer_video.start(30)  # 30ms = 约33fps
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
                self.is_camera_open = True
                print("摄像头已打开")
            else:
                self.stop_camera()
                
        except Exception as e:
            print(f"打开摄像头时出错: {str(e)}")
            QtWidgets.QMessageBox.warning(
                self, u"错误", f"摄像头操作出错: {str(e)}",
                buttons=QtWidgets.QMessageBox.Ok,
                defaultButton=QtWidgets.QMessageBox.Ok)
            self.stop_camera()

    def stop_camera(self):
        """安全关闭摄像头"""
        try:
            self.timer_video.stop()
            if self.cap is not None:
                if self.cap.isOpened():
                    self.cap.release()
                self.cap = None
            if self.out is not None:
                self.out.release()
                self.out = None
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")
            self.is_camera_open = False
            self.frame_count = 0
            print("摄像头已关闭")
        except Exception as e:
            print(f"关闭摄像头时出错: {str(e)}")

    def show_video_frame(self):
        try:
            if self.cap is None or not self.cap.isOpened() or not self.is_camera_open:
                return

            # 检查帧数限制
            if self.frame_count >= self.max_frames:
                print("达到最大帧数限制，重置摄像头")
                self.stop_camera()
                self.button_camera_open()
                return

            name_list = []
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("无法获取摄像头画面")
                self.stop_camera()
                return

            self.frame_count += 1
            showimg = frame.copy()

            with torch.no_grad():
                # 预处理图像
                img = letterbox(showimg, new_shape=self.imgsz)[0]
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
                            plot_one_box(
                                xyxy, showimg, label=label,
                                color=self.colors[int(cls)], line_thickness=2)

            # 保存视频帧
            if self.out is not None:
                self.out.write(showimg)

            # 显示处理后的图像
            label_size = self.label.size()
            img_height, img_width = showimg.shape[:2]
            scale = min(label_size.width() / img_width, label_size.height() / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            show = cv2.resize(showimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                   QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        except Exception as e:
            print(f"处理视频帧时出错: {str(e)}")
            self.stop_camera()

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
    ui.setWindowOpacity(0.93)
    # 去除顶部边框
    # ui.setWindowFlags(Qt.FramelessWindowHint)
    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon.ico"), QIcon.Normal, QIcon.Off)
    # 设置应用图标
    ui.setWindowIcon(icon)
    ui.show()
    sys.exit(app.exec_())
