from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Import models
from models import YOLOv3, ResNet50KeypointModel, GraphTransformer
from feature_engineering import extract_features

class RecognitionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: 'Microsoft YaHei';
                font-size: 30px;
            }
            QLabel {
                font-family: 'Microsoft YaHei';
                font-size: 30px;
            }
            QPushButton {
                font-family: 'Microsoft YaHei';
                font-size: 30px;
            }
            QPushButton#normalize_button {
                background-color: #F5DEB3;  /* beige */
                border-radius: 10px;  /* rounded corners */
                padding: 10px;  /* padding */
            }
            QTextEdit {
                font-family: 'Microsoft YaHei';
                font-size: 30px;
                border: 1px solid #cccccc;
                border-radius: 10px;
                padding: 10px;
                background-color: #f8f8f8;
            }
        """)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setSpacing(40)  # increase spacing
        layout.setContentsMargins(40, 40, 40, 40)  # increase margins

        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 4px solid #cccccc;  # increase border thickness
                border-radius: 20px;  # rounded corners
                background-color: black;
            }
        """)
        self.camera_label.setScaledContents(True)
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label, 2)  # Make camera_label take 2/3 of the space

        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)

        self.result_label = QLabel("Recognition Result")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                border-radius: 20px;  # rounded corners
                padding: 20px;  # padding
                font-size: 30px;
                color: #333333;
            }
        """)
        right_layout.addWidget(self.result_label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter text here...")
        right_layout.addWidget(self.text_edit)

        self.normalize_button = QPushButton("Normalization")
        self.normalize_button.setObjectName("normalize_button")  # Set object name
        self.normalize_button.clicked.connect(self.normalize_text)
        right_layout.addWidget(self.normalize_button)

        self.normalized_text_edit = QTextEdit()
        self.normalized_text_edit.setReadOnly(True)
        self.normalized_text_edit.setFixedHeight(self.text_edit.height())  # Set same height as the above text box
        right_layout.addWidget(self.normalized_text_edit)

        layout.addLayout(right_layout, 1)  # Make right_layout take 1/3 of the space

        self.setLayout(layout)

        # Load models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLOv3().to(self.device)
        self.yolo_model.load_state_dict(torch.load('models/yolov3.pth', map_location=self.device))
        self.yolo_model.eval()
        
        self.keypoint_model = ResNet50KeypointModel().to(self.device)
        self.keypoint_model.load_state_dict(torch.load('models/proposed_ResNet50_finetuned.pth', map_location=self.device))
        self.keypoint_model.eval()

        self.graph_transformer = GraphTransformer(num_classes=34).to(self.device)
        self.graph_transformer.load_state_dict(torch.load('models/graph_transformer.pth', map_location=self.device))
        self.graph_transformer.eval()

        self.previous_gesture = None
        self.current_text = ""

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def detect_hand(self, frame):
        # Detect hand in the frame using YOLO model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((416, 416)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_img = transform(Image.fromarray(frame)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            detections = self.yolo_model(input_img)
        return detections

    def detect_keypoints(self, hand_img):
        # Detect keypoints in the hand image using ResNet50 model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_img = transform(Image.fromarray(hand_img)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            keypoints = self.keypoint_model(input_img)
        return keypoints

    def recognize_gesture(self, keypoints):
        # Recognize gesture from keypoints using Graph Transformer
        keypoints = keypoints.view(1, -1)
        keypoints = keypoints.to(self.device)
        with torch.no_grad():
            output = self.graph_transformer(keypoints)
        predicted = torch.argmax(output, dim=1)
        return predicted.item()

    def update_frame(self):
        # Update frame from the camera
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detect_hand(frame_rgb)
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                hand_img = frame_rgb[int(y1):int(y2), int(x1):int(x2)]
                keypoints = self.detect_keypoints(hand_img)
                current_gesture = self.recognize_gesture(keypoints)
                if self.previous_gesture != current_gesture:
                    self.current_text += chr(current_gesture + 97)  # assuming classes are 'a' to 'z'
                    self.text_edit.setText(self.current_text)
                self.previous_gesture = current_gesture
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_image))

    def normalize_text(self):
        # Normalize text by converting to lowercase and removing spaces
        normalized_text = self.current_text.lower().replace(" ", "")
        self.normalized_text_edit.setText(normalized_text)

    def closeEvent(self, event):
        # Release camera resource when the widget is closed
        self.cap.release()
        super().closeEvent(event)
