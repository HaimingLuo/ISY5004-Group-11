import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
import cv2
from tabs.recognition_tab import RecognitionTab
from tabs.learning_tab import LearningTab
from tabs.input_tab import InputTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition System")
        self.setGeometry(100, 100, 1920, 1080)
        
        # Set the stylesheet for the main window and its components
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: none;
            }
            QTabBar {
                qproperty-drawBase: 0;
                qproperty-expanding: 1;
                alignment: center;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 15px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-size: 30px;
                min-width: 300px;
                min-height: 45px;
                font-family: 'Microsoft YaHei', sans-serif;
            }
            QTabBar::tab:selected {
                background-color: #a9a9a9;
            }
            QLabel {
                font-size: 30px;
                color: #333333;
                font-family: 'Microsoft YaHei', sans-serif;
            }
            QProgressBar {
                font-size: 30px;
                border-radius: 10px;
                text-align: center;
                background-color: #e0e0e0;
                font-family: 'Microsoft YaHei', sans-serif;
            }
        """)
        self.init_ui()

    def init_ui(self):
        # Create the central widget and set it as the central widget of the main window
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a tab widget to hold different tabs
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.North)
        tab_widget.setStyleSheet("""
            QTabWidget::tab-bar {
                alignment: center;
            }
        """)
        
        # Add the different tabs to the tab widget
        tab_widget.addTab(RecognitionTab(), "Recognition")
        tab_widget.addTab(LearningTab(), "Learning")
        tab_widget.addTab(InputTab(), "Input")
        
        # Create a layout for the central widget and add the tab widget to it
        layout = QVBoxLayout(central_widget)
        layout.addWidget(tab_widget)
        layout.setAlignment(Qt.AlignTop)  # Align the tab widget at the top

if __name__ == "__main__":
    # Create an application instance and run the main window
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
