"""
Professional PyQt5 GUI for NBA Player Scoring Predictor
"""
import sys
import os
from typing import Dict, List, Optional
import traceback
import threading
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QPushButton, QComboBox, QSlider, QTextEdit,
    QProgressBar, QCheckBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QGroupBox, QGridLayout, QMessageBox, QFileDialog,
    QStatusBar, QSpacerItem, QSizePolicy, QScrollArea, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPalette, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np

try:
    import qdarkstyle
    DARK_STYLE_AVAILABLE = True
except ImportError:
    DARK_STYLE_AVAILABLE = False
    print("qdarkstyle not available, using default theme")

try:
    from src.predictor import NBAPlayerScoringPredictor
    from utils.logger import main_logger as logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TrainingWorker(QThread):
    """Worker thread for model training to prevent UI freezing."""
    
    progress_updated = pyqtSignal(int)  # Progress percentage
    status_updated = pyqtSignal(str)    # Status message
    log_updated = pyqtSignal(str)       # Log message
    training_completed = pyqtSignal(dict)  # Training results
    training_failed = pyqtSignal(str)   # Error message
    
    def __init__(self, predictor, player_names, optimize, use_cache):
        super().__init__()
        self.predictor = predictor
        self.player_names = player_names
        self.optimize = optimize
        self.use_cache = use_cache
    
    def run(self):
        """Run training in background thread."""
        try:
            # Data collection
            self.status_updated.emit("Collecting data...")
            self.progress_updated.emit(10)
            
            data = self.predictor.collect_data(
                player_names=self.player_names,
                use_cache=self.use_cache
            )
            
            self.log_updated.emit(f"Collected data for {data['PLAYER_NAME'].nunique()} players")
            self.progress_updated.emit(30)
            
            # Feature engineering
            self.status_updated.emit("Engineering features...")
            processed_data = self.predictor.process_data(data)
            
            self.log_updated.emit(f"Engineered {processed_data.shape[1]} features")
            self.progress_updated.emit(60)
            
            # Model training
            self.status_updated.emit("Training models...")
            results = self.predictor.train(processed_data, optimize=self.optimize)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Training completed!")
            
            # Log results
            for model_name, metrics in results.items():
                self.log_updated.emit(
                    f"{model_name}: MAE={metrics['test_mae']:.3f}, R²={metrics['test_r2']:.3f}"
                )
            
            self.training_completed.emit(results)
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.training_failed.emit(error_msg)

class MatplotlibWidget(QWidget):
    """Custom widget for embedding matplotlib plots."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def clear(self):
        """Clear the current plot."""
        self.figure.clear()
        self.canvas.draw()
    
    def plot_model_performance(self, performance_df):
        """Plot model performance comparison."""
        self.figure.clear()
        
        # Create subplots
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        # Set dark theme colors
        self.figure.patch.set_facecolor('#2b2b2b')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#3c3c3c')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Colors for bars
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        # MAE plot
        bars1 = ax1.bar(performance_df['Model'], performance_df['Test MAE'], color=colors)
        ax1.set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('MAE (Points)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', color='white')
        
        # R² plot
        bars2 = ax2.bar(performance_df['Model'], performance_df['Test R²'], color=colors)
        ax2.set_title('R² Score (Higher is Better)', fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', color='white')
        
        self.figure.suptitle('Model Performance Comparison', 
                           fontsize=16, fontweight='bold', color='white')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_feature_importance(self, importance_df):
        """Plot feature importance."""
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#3c3c3c')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        # Create horizontal bar plot
        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['importance'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Feature Importances (XGBoost Model)', fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', color='white')
        
        self.figure.tight_layout()
        self.canvas.draw()

class NBAPlayerScoringGUI(QMainWindow):
    """Professional PyQt5 GUI application."""
    
    def __init__(self):
        super().__init__()
        self.predictor = NBAPlayerScoringPredictor()
        self.is_model_loaded = False
        self.training_worker = None
        
        self.init_ui()
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_periodically)
        self.status_timer.start(5000)  # Update every 5 seconds
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🏀 NBA Player Scoring Predictor - Professional Edition")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create UI components
        self.create_header(main_layout)
        self.create_tabs(main_layout)
        self.create_status_bar()
        
        # Update initial state
        self.update_ui_state()
    
    def create_header(self, layout):
        """Create the header section."""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_frame.setMaximumHeight(100)
        
        header_layout = QHBoxLayout(header_frame)
        
        # Title section
        title_layout = QVBoxLayout()
        
        title_label = QLabel("🏀 NBA Player Scoring Predictor")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setStyleSheet("color: #3498db; margin: 10px;")
        
        subtitle_label = QLabel("Professional Machine Learning System for Basketball Analytics")
        subtitle_label.setFont(QFont("Arial", 10))
        subtitle_label.setStyleSheet("color: #7f8c8d; margin-left: 10px;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        self.load_button = QPushButton("📁 Load Model")
        self.load_button.setMinimumSize(120, 35)
        self.load_button.clicked.connect(self.load_model)
        
        self.train_button = QPushButton("🚀 Train New Model")
        self.train_button.setMinimumSize(140, 35)
        self.train_button.clicked.connect(self.show_training_tab)
        
        self.save_button = QPushButton("💾 Save Model")
        self.save_button.setMinimumSize(120, 35)
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        
        for button in [self.load_button, self.train_button, self.save_button]:
            button.setFont(QFont("Arial", 10, QFont.Bold))
            button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:disabled {
                    background-color: #7f8c8d;
                }
            """)
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.save_button)
        
        # Combine layouts
        header_layout.addLayout(title_layout)
        header_layout.addLayout(button_layout)
        
        layout.addWidget(header_frame)
    
    def create_tabs(self, layout):
        """Create the main tab widget."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Arial", 10))
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #3c3c3c;
            }
            QTabBar::tab {
                background-color: #404040;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
            }
        """)
        
        # Create tabs
        self.create_prediction_tab()
        self.create_analysis_tab()
        self.create_training_tab()
        
        layout.addWidget(self.tab_widget)
    
    def create_prediction_tab(self):
        """Create the prediction tab."""
        prediction_widget = QWidget()
        layout = QHBoxLayout(prediction_widget)
        
        # Left panel - Controls
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_frame.setMaximumWidth(350)
        control_frame.setMinimumWidth(300)
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #404040;
                border-radius: 5px;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        
        # Player selection group
        player_group = QGroupBox("Player Selection")
        player_group.setFont(QFont("Arial", 11, QFont.Bold))
        player_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        player_layout = QVBoxLayout(player_group)
        
        player_layout.addWidget(QLabel("Select Player:"))
        
        self.player_combo = QComboBox()
        self.player_combo.setMinimumHeight(30)
        self.player_combo.addItem("Load model first...")
        self.player_combo.setEnabled(False)
        self.player_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2c3e50;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }
        """)
        player_layout.addWidget(self.player_combo)
        
        refresh_button = QPushButton("🔄 Refresh Players")
        refresh_button.setMinimumHeight(35)
        refresh_button.clicked.connect(self.refresh_players)
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        player_layout.addWidget(refresh_button)
        
        control_layout.addWidget(player_group)
        
        # Analysis settings group
        settings_group = QGroupBox("Analysis Settings")
        settings_group.setFont(QFont("Arial", 11, QFont.Bold))
        settings_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        settings_layout = QVBoxLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Recent Games for Analysis:"))
        
        self.recent_games_slider = QSlider(Qt.Horizontal)
        self.recent_games_slider.setMinimum(5)
        self.recent_games_slider.setMaximum(20)
        self.recent_games_slider.setValue(10)
        self.recent_games_slider.valueChanged.connect(self.update_recent_games_label)
        self.recent_games_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #555;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        settings_layout.addWidget(self.recent_games_slider)
        
        self.recent_games_label = QLabel("10 games")
        self.recent_games_label.setAlignment(Qt.AlignCenter)
        self.recent_games_label.setStyleSheet("color: white;")
        settings_layout.addWidget(self.recent_games_label)
        
        control_layout.addWidget(settings_group)
        
        # Predict button
        self.predict_button = QPushButton("🎯 PREDICT POINTS")
        self.predict_button.setMinimumHeight(50)
        self.predict_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.predict_button.clicked.connect(self.predict_player_points)
        self.predict_button.setEnabled(False)
        
        control_layout.addWidget(self.predict_button)
        control_layout.addStretch()
        
        # Right panel - Results
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.StyledPanel)
        results_frame.setStyleSheet("""
            QFrame {
                background-color: #404040;
                border-radius: 5px;
            }
        """)
        
        results_layout = QVBoxLayout(results_frame)
        
        results_header = QLabel("🔮 Prediction Results")
        results_header.setFont(QFont("Arial", 16, QFont.Bold))
        results_header.setAlignment(Qt.AlignCenter)
        results_header.setStyleSheet("color: #3498db; margin: 10px;")
        results_layout.addWidget(results_header)
        
        self.results_text = QTextEdit()
        self.results_text.setFont(QFont("Consolas", 10))
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 10px;
            }
        """)
        self.results_text.setPlainText("Load a trained model and select a player to see predictions...")
        results_layout.addWidget(self.results_text)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_frame)
        splitter.addWidget(results_frame)
        splitter.setSizes([350, 850])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(prediction_widget, "🎯 Predictions")
    
    def create_analysis_tab(self):
        """Create the analysis tab."""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        performance_button = QPushButton("📊 Model Performance")
        performance_button.setMinimumHeight(40)
        performance_button.clicked.connect(self.show_model_performance)
        performance_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        importance_button = QPushButton("🎯 Feature Importance")
        importance_button.setMinimumHeight(40)
        importance_button.clicked.connect(self.show_feature_importance)
        importance_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        button_layout.addWidget(performance_button)
        button_layout.addWidget(importance_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Plot area
        self.plot_widget = MatplotlibWidget()
        layout.addWidget(self.plot_widget)
        
        self.tab_widget.addTab(analysis_widget, "📈 Analysis")
    
    def create_training_tab(self):
        """Create the training tab."""
        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)
        
        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_group.setFont(QFont("Arial", 12, QFont.Bold))
        config_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        config_layout = QGridLayout(config_group)
        
        # Player selection
        config_layout.addWidget(QLabel("Players (leave empty for all):"), 0, 0)
        self.training_players_entry = QLineEdit()
        self.training_players_entry.setPlaceholderText("e.g., LeBron James, Stephen Curry, Luka Dončić")
        self.training_players_entry.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2c3e50;
                color: white;
            }
        """)
        config_layout.addWidget(self.training_players_entry, 0, 1)
        
        # Options
        self.optimize_checkbox = QCheckBox("Optimize hyperparameters (slower but better)")
        self.optimize_checkbox.setStyleSheet("color: white;")
        self.use_cache_checkbox = QCheckBox("Use cached data when available")
        self.use_cache_checkbox.setChecked(True)
        self.use_cache_checkbox.setStyleSheet("color: white;")
        
        config_layout.addWidget(self.optimize_checkbox, 1, 0, 1, 2)
        config_layout.addWidget(self.use_cache_checkbox, 2, 0, 1, 2)
        
        layout.addWidget(config_group)
        
        # Training control
        control_layout = QHBoxLayout()
        
        self.start_training_button = QPushButton("🚀 START TRAINING")
        self.start_training_button.setMinimumHeight(50)
        self.start_training_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.start_training_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 10px 30px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.start_training_button.clicked.connect(self.start_training)
        
        self.stop_training_button = QPushButton("⏹️ STOP TRAINING")
        self.stop_training_button.setMinimumHeight(50)
        self.stop_training_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.stop_training_button.setEnabled(False)
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 10px 30px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        
        control_layout.addWidget(self.start_training_button)
        control_layout.addWidget(self.stop_training_button)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_label = QLabel("Ready to train")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: white; font-weight: bold;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        log_layout = QVBoxLayout(log_group)
        
        self.training_log = QTextEdit()
        self.training_log.setFont(QFont("Consolas", 9))
        self.training_log
        self.training_log.setMaximumHeight(200)
        self.training_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.training_log)
        
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(training_widget, "🏋️ Training")
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets to status bar
        self.model_status_label = QLabel("Model: Not Loaded")
        self.model_status_label.setStyleSheet("color: #e74c3c;")
        
        self.data_status_label = QLabel("Data: Ready")
        self.data_status_label.setStyleSheet("color: #27ae60;")
        
        self.status_bar.addWidget(self.model_status_label)
        self.status_bar.addPermanentWidget(self.data_status_label)
        
        self.update_status("Ready - Load or train a model to begin predictions")
    
    def update_status(self, message):
        """Update status bar message."""
        self.status_bar.showMessage(f"Status: {message}")
    
    def update_status_periodically(self):
        """Periodic status updates."""
        if hasattr(self, 'predictor') and self.predictor:
            try:
                # Update data status
                players = self.predictor.get_available_players()
                self.data_status_label.setText(f"Data: {len(players)} players cached")
            except:
                self.data_status_label.setText("Data: Ready")
    
    def update_ui_state(self):
        """Update UI state based on model status."""
        self.predict_button.setEnabled(self.is_model_loaded)
        self.save_button.setEnabled(self.is_model_loaded)
        self.player_combo.setEnabled(self.is_model_loaded)
        
        if self.is_model_loaded:
            self.model_status_label.setText("Model: Loaded ✓")
            self.model_status_label.setStyleSheet("color: #27ae60;")
        else:
            self.model_status_label.setText("Model: Not Loaded")
            self.model_status_label.setStyleSheet("color: #e74c3c;")
    
    def update_recent_games_label(self, value):
        """Update recent games label."""
        self.recent_games_label.setText(f"{value} games")
    
    def show_training_tab(self):
        """Switch to training tab."""
        self.tab_widget.setCurrentIndex(2)
    
    def load_model(self):
        """Load a trained model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Trained Model", "", "Pickle files (*.pkl);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.update_status("Loading model...")
                QApplication.processEvents()
                
                self.predictor.load_model(file_path)
                self.is_model_loaded = True
                
                self.update_ui_state()
                self.refresh_players()
                
                self.update_status("Model loaded successfully!")
                QMessageBox.information(self, "Success", "Model loaded successfully!")
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
                self.update_status("Error loading model")
    
    def save_model(self):
        """Save the current trained model."""
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Warning", "No trained model to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trained Model", "nba_model.pkl", "Pickle files (*.pkl);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.update_status("Saving model...")
                QApplication.processEvents()
                
                self.predictor.save_model(file_path)
                self.update_status("Model saved successfully!")
                QMessageBox.information(self, "Success", f"Model saved to {file_path}")
                
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")
                self.update_status("Error saving model")
    
    def refresh_players(self):
        """Refresh the player dropdown list."""
        try:
            if self.is_model_loaded:
                players = self.predictor.get_available_players()
                
                self.player_combo.clear()
                if players:
                    self.player_combo.addItems(players)
                else:
                    self.player_combo.addItem("No players found")
                    self.player_combo.setEnabled(False)
            else:
                self.player_combo.clear()
                self.player_combo.addItem("Load model first...")
                self.player_combo.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Error refreshing players: {e}")
            self.player_combo.clear()
            self.player_combo.addItem("Error loading players")
            self.player_combo.setEnabled(False)
    
    def predict_player_points(self):
        """Predict points for the selected player."""
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a trained model first.")
            return
        
        player_name = self.player_combo.currentText()
        if not player_name or player_name in ["Load model first...", "No players found"]:
            QMessageBox.warning(self, "Warning", "Please select a valid player.")
            return
        
        recent_games = self.recent_games_slider.value()
        
        try:
            self.update_status(f"Predicting points for {player_name}...")
            self.results_text.setPlainText("Generating predictions...\n")
            QApplication.processEvents()
            
            # Make prediction
            predictions = self.predictor.predict_player_points(player_name, recent_games)
            
            # Display results
            self.display_prediction_results(predictions)
            self.update_status("Prediction completed successfully!")
            
        except Exception as e:
            error_msg = f"Error predicting for {player_name}: {str(e)}"
            logger.error(error_msg)
            self.results_text.setPlainText(f"Error: {error_msg}")
            self.update_status("Prediction failed")
    
    def display_prediction_results(self, predictions: Dict):
        """Display prediction results in the text widget."""
        player_name = predictions.get('player_name', 'Unknown Player')
        recent_avg = predictions.get('recent_average', 0)
        
        # Build result text
        result_text = f"🏀 PREDICTION RESULTS FOR {player_name.upper()}\n"
        result_text += "=" * 60 + "\n\n"
        
        # Recent performance context
        recent_games = self.recent_games_slider.value()
        result_text += f"📊 Recent Average ({recent_games} games): {recent_avg:.1f} points\n\n"
        
        # Model predictions
        result_text += "🤖 MODEL PREDICTIONS:\n"
        result_text += "-" * 40 + "\n"
        
        model_order = ['ensemble', 'xgboost', 'lightgbm', 'random_forest', 'neural_network']
        
        for model_name in model_order:
            if model_name in predictions:
                pred_data = predictions[model_name]
                pred_points = pred_data['predicted_points']
                ci_low, ci_high = pred_data['confidence_interval']
                mae = pred_data['model_mae']
                
                result_text += f"\n{model_name.replace('_', ' ').title()}:\n"
                result_text += f"  • Predicted Points: {pred_points:.1f}\n"
                result_text += f"  • Range: {ci_low:.1f} - {ci_high:.1f} points\n"
                result_text += f"  • Model Accuracy (MAE): ±{mae:.1f} points\n"
        
        # Analysis
        ensemble_pred = predictions.get('ensemble', {}).get('predicted_points', 0)
        diff_from_avg = ensemble_pred - recent_avg
        
        result_text += "\n" + "=" * 60 + "\n"
        result_text += "📈 ANALYSIS:\n"
        result_text += "-" * 40 + "\n"
        
        if abs(diff_from_avg) < 2:
            trend = "consistent with"
        elif diff_from_avg > 2:
            trend = "above"
        else:
            trend = "below"
        
        result_text += f"• Ensemble prediction is {trend} recent average\n"
        result_text += f"• Difference from recent avg: {diff_from_avg:+.1f} points\n"
        
        if ensemble_pred > recent_avg:
            result_text += "• 📈 Model suggests potential for increased scoring\n"
        elif ensemble_pred < recent_avg:
            result_text += "• 📉 Model suggests potential for decreased scoring\n"
        else:
            result_text += "• ➡️ Model suggests consistent performance\n"
        
        # Confidence assessment
        ensemble_mae = predictions.get('ensemble', {}).get('model_mae', 5)
        if ensemble_mae < 4:
            confidence = "HIGH"
        elif ensemble_mae < 6:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        result_text += f"• Prediction Confidence: {confidence}\n"
        
        # Fantasy/Betting Insights
        result_text += "\n" + "🎯 FANTASY/BETTING INSIGHTS:\n"
        result_text += "-" * 40 + "\n"
        
        if confidence == "HIGH" and ensemble_pred > recent_avg + 2:
            result_text += "• 🔥 STRONG BUY: Model confident in over-performance\n"
        elif confidence == "HIGH" and ensemble_pred < recent_avg - 2:
            result_text += "• 🧊 FADE PLAY: Model confident in under-performance\n"
        elif confidence == "MEDIUM":
            result_text += "• ⚖️ NEUTRAL: Moderate confidence, proceed with caution\n"
        else:
            result_text += "• ❓ LOW CONFIDENCE: High variance expected\n"
        
        # Disclaimer
        result_text += "\n" + "⚠️  DISCLAIMER:\n"
        result_text += "Predictions based on historical performance and statistical models.\n"
        result_text += "Actual results may vary due to injuries, matchups, and other factors.\n"
        result_text += "Use for entertainment and analysis purposes only.\n"
        
        self.results_text.setPlainText(result_text)
    
    def start_training(self):
        """Start model training."""
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.warning(self, "Warning", "Training is already in progress.")
            return
        
        # Get parameters
        players_text = self.training_players_entry.text().strip()
        player_names = [name.strip() for name in players_text.split(',')] if players_text else None
        optimize = self.optimize_checkbox.isChecked()
        use_cache = self.use_cache_checkbox.isChecked()
        
        # Confirm training start
        if optimize:
            reply = QMessageBox.question(
                self, "Confirm Training",
                "Hyperparameter optimization is enabled. This may take 30+ minutes.\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # Update UI
        self.start_training_button.setEnabled(False)
        self.stop_training_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.training_log.clear()
        
        # Create and start worker
        self.training_worker = TrainingWorker(self.predictor, player_names, optimize, use_cache)
        
        # Connect signals
        self.training_worker.progress_updated.connect(self.progress_bar.setValue)
        self.training_worker.status_updated.connect(self.progress_label.setText)
        self.training_worker.log_updated.connect(self.add_training_log)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.training_failed.connect(self.on_training_failed)
        
        # Start training
        self.training_worker.start()
        self.update_status("Training started...")
    
    def stop_training(self):
        """Stop model training."""
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Stop",
                "Are you sure you want to stop training?\nProgress will be lost.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_worker.terminate()
                self.training_worker.wait()
                self.on_training_stopped()
    
    def on_training_completed(self, results):
        """Handle training completion."""
        self.is_model_loaded = True
        self.update_ui_state()
        self.refresh_players()
        
        # Reset training UI
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        
        self.update_status("Training completed successfully!")
        
        # Show results
        best_model = min(results.keys(), key=lambda k: results[k]['test_mae'])
        best_mae = results[best_model]['test_mae']
        
        QMessageBox.information(
            self, "Training Complete",
            f"Model training completed successfully!\n\n"
            f"Best Model: {best_model.title()}\n"
            f"Best MAE: {best_mae:.3f} points"
        )
    
    def on_training_failed(self, error_message):
        """Handle training failure."""
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        
        self.update_status("Training failed")
        QMessageBox.critical(self, "Training Failed", f"Training failed:\n{error_message}")
    
    def on_training_stopped(self):
        """Handle training stop."""
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        self.progress_label.setText("Training stopped")
        self.update_status("Training stopped by user")
    
    def add_training_log(self, message):
        """Add message to training log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.training_log.textCursor()
        cursor.movePosition(cursor.End)
        self.training_log.setTextCursor(cursor)
    
    def show_model_performance(self):
        """Show model performance analysis."""
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Warning", "Please load or train a model first.")
            return
        
        try:
            performance_df = self.predictor.get_model_performance()
            self.plot_widget.plot_model_performance(performance_df)
            self.tab_widget.setCurrentIndex(1)  # Switch to analysis tab
            
        except Exception as e:
            logger.error(f"Error showing model performance: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show performance:\n{str(e)}")
    
    def show_feature_importance(self):
        """Show feature importance analysis."""
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Warning", "Please load or train a model first.")
            return
        
        try:
            importance_df = self.predictor.get_feature_importance(top_n=15)
            
            if importance_df.empty:
                QMessageBox.information(self, "Info", "Feature importance not available for this model.")
                return
            
            self.plot_widget.plot_feature_importance(importance_df)
            self.tab_widget.setCurrentIndex(1)  # Switch to analysis tab
            
        except Exception as e:
            logger.error(f"Error showing feature importance: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show feature importance:\n{str(e)}")
    
    def closeEvent(self, event):
        """Handle application close event."""
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Training is in progress. Are you sure you want to exit?\nProgress will be lost.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_worker.terminate()
                self.training_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main function to run the PyQt5 GUI application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("NBA Player Scoring Predictor")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Basketball Analytics")
    
    # Apply dark theme if available
    if DARK_STYLE_AVAILABLE:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    else:
        # Basic dark theme fallback
        app.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: white;
            }
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)
    
    # Create and show main window
    window = NBAPlayerScoringGUI()
    window.show()
    
    # Center window on screen
    screen = app.primaryScreen().geometry()
    window.move(
        (screen.width() - window.width()) // 2,
        (screen.height() - window.height()) // 2
    )
    
    logger.info("NBA Player Scoring Predictor GUI started")
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()