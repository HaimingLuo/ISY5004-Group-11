from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QLinearGradient, QBrush, QPalette, QColor
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Import models
from models import YOLOv3, ResNet50KeypointModel, SiameseMLP
from feature_engineering import extract_features

class LearningTab(QWidget):
    def __init__(self):
        super().__init__()
        self.set_background_gradient()  # Set background gradient
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
            }
            QLabel {
                font-size: 30px;
                color: #333333;
                font-family: 'Microsoft YaHei', sans-serif;
            }
            QComboBox {
                font-size: 30px;
                padding: 15px;
                min-height: 80px;
                min-width: 200px;
                border: 2px solid #cccccc;
                border-radius: 20px;
                background-color: #f0f0f0;
                font-family: 'Microsoft YaHei', sans-serif;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 1px;
                border-left-color: #cccccc;
                border-left-style: solid;
                border-top-right-radius: 20px;
                border-bottom-right-radius: 20px;
                background-color: #e0e0e0;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 20px;
                height: 20px;
            }
        """)
        self.score_counter = 0
        self.init_ui()  # Initialize the UI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000)

    def set_background_gradient(self):
        # Set a vertical gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#e0f7fa"))
        gradient.setColorAt(1.0, QColor("#b2ebf2"))
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

    def init_ui(self):
        # Create main layout
        layout = QHBoxLayout()
        layout.setSpacing(40)
        layout.setContentsMargins(40, 40, 40, 40)

        # Add camera label to the layout
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 4px solid #cccccc;
                border-radius: 20px;
                background-color: black;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
        """)
        self.camera_label.setScaledContents(True)
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label, 2)

        # Add a right-side layout with combo box and image labels
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)

        self.combo_box = QComboBox()
        self.combo_box.addItems([chr(i) for i in range(97, 123)] + [str(i) for i in range(10)])
        self.combo_box.currentIndexChanged.connect(self.update_image)
        right_layout.addWidget(self.combo_box)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                font-family: 'Microsoft YaHei', sans-serif;
            }
        """)
        right_layout.addWidget(self.image_label)

        self.similarity_image_label = QLabel()
        self.similarity_image_label.setAlignment(Qt.AlignCenter)
        self.similarity_image_label.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                border-radius: 20px;
                padding: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                font-family: 'Microsoft YaHei', sans-serif;
            }
        """)
        right_layout.addWidget(self.similarity_image_label)

        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("""
            QLabel {
                font-size: 30px;
                color: #333333;
                font-family: 'Microsoft YaHei', sans-serif;
            }
        """)
        right_layout.addWidget(self.progress_label)

        layout.addLayout(right_layout, 1)

        self.setLayout(layout)

        # Load models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLOv3().to(self.device)
        self.yolo_model.load_state_dict(torch.load('models/yolov3.pth', map_location=self.device))
        self.yolo_model.eval()

        self.keypoint_model = ResNet50KeypointModel().to(self.device)
        self.keypoint_model.load_state_dict(torch.load('models/proposed_ResNet50_finetuned.pth', map_location=self.device))
        self.keypoint_model.eval()

        self.siamese_model = SiameseMLP(input_dim=128, hidden_dim=128, output_dim=64).to(self.device)
        self.siamese_model.load_state_dict(torch.load('models/siamese_mlp.pth', map_location=self.device))
        self.siamese_model.eval()

        self.prototypes = self.load_prototypes('prototypes.txt')
        self.previous_gesture = None

    def load_prototypes(self, file_path):
        # Load prototypes from file
        prototypes = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = parts[0]
                prototype = torch.tensor(list(map(float, parts[1:])), dtype=torch.float32)
                prototypes[label] = prototype
        return prototypes

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
        # Recognize gesture from keypoints using Siamese MLP
        keypoints = keypoints.view(1, -1)
        keypoints = keypoints.to(self.device)
        with torch.no_grad():
            embedding = self.siamese_model(keypoints).cpu().numpy().reshape(1, -1)
        
        best_label = None
        max_similarity = -1
        for label, prototype in self.prototypes.items():
            prototype = prototype.numpy().reshape(1, -1)
            similarity = cosine_similarity(embedding, prototype)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_label = label
        
        return best_label, max_similarity

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
                current_gesture, similarity = self.recognize_gesture(keypoints)
                if self.previous_gesture != current_gesture and similarity > 0.7:
                    selected_class = self.combo_box.currentText()
                    if selected_class == current_gesture:
                        self.score_counter += 1
                    else:
                        self.score_counter = 0

                    if self.score_counter >= 10:
                        check_image_path = os.path.join(os.path.dirname(__file__), "images", "check.png")
                        if os.path.exists(check_image_path):
                            pixmap = QPixmap(check_image_path)
                            self.progress_label.setPixmap(pixmap.scaled(self.progress_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        self.progress_label.setText("Not perfect yet, keep trying")

                    self.previous_gesture = current_gesture

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_image))

    def update_image(self):
        # Update displayed image based on combo box selection
        selection = self.combo_box.currentText()
        image_path = os.path.join(os.path.dirname(__file__), "picture", f"{selection}.jpg")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.setText("Image not found")
