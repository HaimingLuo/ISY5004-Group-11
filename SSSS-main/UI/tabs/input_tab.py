from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar, QMessageBox, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
from torchvision import transforms
from PIL import Image
import os

from models import SiameseMLP
from feature_engineering import extract_features

class ProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal(np.ndarray)

    def __init__(self, file_paths, model):
        super().__init__()
        self.file_paths = file_paths
        self.model = model

    def run(self):
        vectors = []
        for i, file_path in enumerate(self.file_paths):
            image = Image.open(file_path)
            image_tensor = self.preprocess_image(image)
            with torch.no_grad():
                embedding = self.model(image_tensor).cpu().numpy().flatten()
            vectors.append(embedding)
            self.update_progress.emit(int((i + 1) / len(self.file_paths) * 100))
        average_vector = np.mean(vectors, axis=0)
        self.processing_complete.emit(average_vector)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.model.device)
        return image_tensor

class InputTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel("Upload multiple images to calculate the average embedding vector")
        layout.addWidget(self.label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter a name for the vector")
        layout.addWidget(self.text_edit)

        self.upload_button = QPushButton("Upload Images")
        self.upload_button.clicked.connect(self.upload_images)
        layout.addWidget(self.upload_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SiameseMLP(input_dim=128, hidden_dim=128, output_dim=64).to(self.device)
        self.model.load_state_dict(torch.load('models/siamese_mlp.pth', map_location=self.device))
        self.model.eval()

    def upload_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.xpm *.jpg)", options=options)
        if file_paths:
            self.process_images(file_paths)

    def process_images(self, file_paths):
        self.progress_bar.setValue(0)
        self.thread = ProcessingThread(file_paths, self.model)
        self.thread.update_progress.connect(self.progress_bar.setValue)
        self.thread.processing_complete.connect(self.save_vector)
        self.thread.start()

    def save_vector(self, average_vector):
        name = self.text_edit.toPlainText().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter a name for the vector.")
            return
        with open('prototypes.txt', 'a') as f:
            vector_str = ' '.join(map(str, average_vector))
            f.write(f'{name} {vector_str}\n')
        QMessageBox.information(self, "Success", "Vector saved successfully.")
        self.reset_ui()

    def reset_ui(self):
        self.text_edit.clear()
        self.progress_bar.setValue(0)
        
