import sys

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# -------------------------- 配置修改处 --------------------------
# 改成你yolov5m训练输出的exp文件夹，比如exp11、exp12
WEIGHT_PATH = "runs/train/exp2/weights/best.pt"
IMG_DISPLAY_WIDTH = 550
# --------------------------------------------------------------------------------


class AnimalDetectGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动物识别检测系统 YOLOv5")
        self.setFixedSize(1200, 700)

        # 加载模型
        print("正在加载训练模型...")
        self.model = torch.hub.load(
            repo_or_dir=".", model="custom", path=WEIGHT_PATH, source="local", force_reload=False
        )
        # 关键修改：置信度提高，过滤低可信度混淆框，解决骆驼识别成牛
        self.model.conf = 0.52
        self.model.iou = 0.3
        print("模型加载完成！")

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 顶部按钮区
        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("选择本地图片")
        self.select_btn.clicked.connect(self.select_image)
        self.select_btn.setFixedHeight(40)
        btn_layout.addWidget(self.select_btn)
        main_layout.addLayout(btn_layout)

        # 2. 图片显示区域（原图 | 识别结果）
        img_layout = QHBoxLayout()
        # 原图
        self.label_origin = QLabel("原始图片")
        self.label_origin.setAlignment(Qt.AlignCenter)
        self.label_origin.setStyleSheet("border:1px solid #999;")
        self.label_origin.setFixedSize(IMG_DISPLAY_WIDTH, 500)
        # 结果图
        self.label_result = QLabel("识别结果图")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("border:1px solid #999;")
        self.label_result.setFixedSize(IMG_DISPLAY_WIDTH, 500)

        img_layout.addWidget(self.label_origin)
        img_layout.addWidget(self.label_result)
        main_layout.addLayout(img_layout)

        # 3. 底部日志输出框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        main_layout.addWidget(self.log_text)

    def select_image(self):
        # 打开文件选择窗口
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return
        self.log_text.append(f"已选择图片：{file_path}")

        # 显示原图
        origin_pix = QPixmap(file_path).scaled(IMG_DISPLAY_WIDTH, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_origin.setPixmap(origin_pix)

        # YOLO推理识别
        results = self.model(file_path)
        # 绘制标注图
        img_rgb = results.render()[0]
        h, w, c = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, c * w, QImage.Format_RGB888)
        result_pix = QPixmap.fromImage(q_img).scaled(
            IMG_DISPLAY_WIDTH, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_result.setPixmap(result_pix)

        # 打印识别信息
        self.log_text.append("===== 识别目标详情 =====")
        df = results.pandas().xyxy[0]
        if len(df) == 0:
            self.log_text.append("未检测到动物")
        else:
            for idx, row in df.iterrows():
                cls_name = row["name"]
                conf = round(row["confidence"], 3)
                self.log_text.append(f"类别：{cls_name} | 置信度：{conf}")
        self.log_text.append("\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnimalDetectGUI()
    window.show()
    sys.exit(app.exec_())
