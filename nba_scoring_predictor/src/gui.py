# -*- coding: utf-8 -*-
"""
Professional PyQt5 GUI for NBA Player Scoring Predictor - FIXED VERSION
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
    
    from src.player_search_widget import PlayerSearchWidget
    from src.predictor import EnhancedNBAPredictor
    from utils.logger import main_logger as logger
    from utils.nba_player_fetcher import NBAPlayerFetcher
    from utils.player_roles import PlayerRoles
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class NBADataUpdateWorker(QThread):
    """Worker thread for updating NBA players data."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    update_completed = pyqtSignal(dict)
    update_failed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.fetcher = NBAPlayerFetcher()
    
    def run(self):
        """Run NBA data update in background thread."""
        try:
            self.status_updated.emit("Connecting to NBA API...")
            self.progress_updated.emit(10)
            
            self.status_updated.emit("Fetching team rosters...")
            self.progress_updated.emit(30)
            
            # Fetch all active players
            data = self.fetcher.fetch_all_active_players()
            
            self.progress_updated.emit(90)
            self.status_updated.emit("Processing player data...")
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Update completed!")
            
            self.update_completed.emit(data)
            
        except Exception as e:
            error_msg = f"NBA data update failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.update_failed.emit(error_msg)


class TrainingWorker(QThread):
    """Enhanced worker thread for model training with role support."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    training_failed = pyqtSignal(str)
    
    def __init__(self, predictor, player_names, roles, optimize, use_cache, role_based):
        super().__init__()
        self.predictor = predictor
        self.player_names = player_names
        self.roles = roles
        self.optimize = optimize
        self.use_cache = use_cache
        self.role_based = role_based
    
    def run(self):
        """Run training in background thread with role support."""
        try:
            # Data collection
            self.status_updated.emit("Collecting data...")
            self.progress_updated.emit(10)
            
            if self.role_based and self.roles:
                # Role-based data collection
                data = self.predictor.collect_data_by_roles(
                    roles=self.roles,
                    max_per_role=3,
                    use_cache=self.use_cache
                )
            else:
                # Regular data collection
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
            
            if self.role_based:
                results = self.predictor.train_with_roles(processed_data, optimize=self.optimize)
            else:
                results = self.predictor.train(processed_data, optimize=self.optimize)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Training completed!")
            
            # Log results
            for model_name, metrics in results.items():
                self.log_updated.emit(
                    f"{model_name}: MAE={metrics['test_mae']:.3f}, R={metrics['test_r2']:.3f}"
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
        
        # R plot
        bars2 = ax2.bar(performance_df['Model'], performance_df['Test R'], color=colors)
        ax2.set_title('R Score (Higher is Better)', fontweight='bold')
        ax2.set_ylabel('R Score')
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
    """Enhanced Professional PyQt5 GUI application - FIXED VERSION."""
    
    def __init__(self):
        super().__init__()
        self.predictor = EnhancedNBAPredictor()
        self.is_model_loaded = False
        self.training_worker = None
        self.nba_update_worker = None
        self.selected_player_name = None
        self.selected_players = []
        
        # Initialize NBA data components
        self.player_fetcher = NBAPlayerFetcher()
        self.player_roles = PlayerRoles()
        
        self.init_ui()
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_periodically)
        self.status_timer.start(5000)
        
        self.ensure_nba_data_available()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🏀 NBA Player Scoring Predictor - Professional Edition v2.0")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1400, 900)
        
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
        self.check_nba_data_status()
    
    def create_header(self, layout):
        """Create the enhanced header section."""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_frame.setMaximumHeight(120)
        
        header_layout = QHBoxLayout(header_frame)
        
        # Title section
        title_layout = QVBoxLayout()
        
        title_label = QLabel("🏀 NBA Player Scoring Predictor")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setStyleSheet("color: #3498db; margin: 10px;")
        
        subtitle_label = QLabel("Professional ML System with Live NBA Data Integration")
        subtitle_label.setFont(QFont("Arial", 11))
        subtitle_label.setStyleSheet("color: #7f8c8d; margin-left: 10px;")
        
        version_label = QLabel("Version 2.0 - Enhanced with Role-Based Training")
        version_label.setFont(QFont("Arial", 9))
        version_label.setStyleSheet("color: #95a5a6; margin-left: 10px;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addWidget(version_label)
        
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
        """Create the enhanced main tab widget."""
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
        self.create_team_comparison_tab()  # ADD THIS LINE!
        self.create_nba_data_tab()
    
        layout.addWidget(self.tab_widget)
    
    def create_prediction_tab(self):
        """Create the enhanced prediction tab with search functionality."""
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
        
        # Player search group
        search_group = QGroupBox("Player Search")
        search_group.setFont(QFont("Arial", 11, QFont.Bold))
        search_group.setStyleSheet("""
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
        
        search_layout = QVBoxLayout(search_group)
        
        # Add the search widget
        self.player_search_widget = PlayerSearchWidget()
        self.player_search_widget.player_selected.connect(self.on_player_search_selected)
        search_layout.addWidget(self.player_search_widget)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #555;")
        search_layout.addWidget(separator)
        
        # Traditional dropdown (as backup)
        dropdown_label = QLabel("Or select from dropdown:")
        dropdown_label.setStyleSheet("color: #95a5a6; font-size: 10px; margin-top: 10px;")
        search_layout.addWidget(dropdown_label)
        
        self.player_combo = QComboBox()
        self.player_combo.setMinimumHeight(30)
        self.player_combo.addItem("Load model first...")
        self.player_combo.setEnabled(False)
        self.player_combo.currentTextChanged.connect(self.on_dropdown_selection_changed)
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
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
            }
        """)
        search_layout.addWidget(self.player_combo)
        
        control_layout.addWidget(search_group)
        
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
        self.results_text.setPlainText("Load a trained model and search for a player to see predictions...")
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
        """Create the enhanced training tab with role-based features."""
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
        
        # Training method selection
        config_layout.addWidget(QLabel("Training Method:"), 0, 0)
        
        self.training_method_combo = QComboBox()
        self.training_method_combo.addItems([
            "Role-Based Training (Recommended)",
            "Select Specific Players",
            "Enter Custom Players",
            "Train All Available Players"
        ])
        self.training_method_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2c3e50;
                color: white;
                min-height: 20px;
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
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
            }
        """)
        self.training_method_combo.currentTextChanged.connect(self.on_training_method_changed)
        config_layout.addWidget(self.training_method_combo, 0, 1)
        
        # Role selection (for role-based training)
        config_layout.addWidget(QLabel("Select Positions:"), 1, 0)
        
        self.role_selection_widget = QWidget()
        role_selection_layout = QVBoxLayout(self.role_selection_widget)
        role_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        # Role checkboxes
        role_checkbox_layout = QHBoxLayout()
        self.role_checkboxes = {}
        roles = ["PG", "SG", "SF", "PF", "C"]
        
        for role in roles:
            checkbox = QCheckBox(role)
            checkbox.setChecked(True)
            checkbox.setStyleSheet("color: white; font-weight: bold;")
            self.role_checkboxes[role] = checkbox
            role_checkbox_layout.addWidget(checkbox)
        
        role_selection_layout.addLayout(role_checkbox_layout)
        
        # Players per role slider
        players_per_role_layout = QHBoxLayout()
        players_per_role_layout.addWidget(QLabel("Max players per position:"))
        
        self.players_per_role_slider = QSlider(Qt.Horizontal)
        self.players_per_role_slider.setMinimum(1)
        self.players_per_role_slider.setMaximum(8)
        self.players_per_role_slider.setValue(3)
        self.players_per_role_slider.valueChanged.connect(self.update_players_per_role_label)
        self.players_per_role_slider.setStyleSheet("""
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
       
        self.players_per_role_label = QLabel("3 players")
        self.players_per_role_label.setStyleSheet("color: white; font-weight: bold;")
       
        players_per_role_layout.addWidget(self.players_per_role_slider)
        players_per_role_layout.addWidget(self.players_per_role_label)
        role_selection_layout.addLayout(players_per_role_layout)
       
        config_layout.addWidget(self.role_selection_widget, 1, 1)
       
        # Custom player entry
        config_layout.addWidget(QLabel("Custom Players:"), 2, 0)
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
        config_layout.addWidget(self.training_players_entry, 2, 1)
       
        # Training options
        self.optimize_checkbox = QCheckBox("Optimize hyperparameters (slower but better)")
        self.optimize_checkbox.setStyleSheet("color: white;")
        self.use_cache_checkbox = QCheckBox("Use cached data when available")
        self.use_cache_checkbox.setChecked(True)
        self.use_cache_checkbox.setStyleSheet("color: white;")
       
        config_layout.addWidget(self.optimize_checkbox, 3, 0, 1, 2)
        config_layout.addWidget(self.use_cache_checkbox, 4, 0, 1, 2)
       
        layout.addWidget(config_group)
       
        # Set initial state
        self.on_training_method_changed()
       
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
   
    def create_nba_data_tab(self):
        """Create the NBA data management tab."""
        nba_data_widget = QWidget()
        layout = QVBoxLayout(nba_data_widget)
       
        # Header
        header_label = QLabel("🏀 NBA Data Management")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("color: #3498db; margin: 10px;")
        layout.addWidget(header_label)
       
        # Data status section
        status_group = QGroupBox("Current Data Status")
        status_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
       
        status_layout = QVBoxLayout(status_group)
       
        self.nba_data_status_label = QLabel("Checking NBA data status...")
        self.nba_data_status_label.setStyleSheet("color: white; font-size: 12px; margin: 10px;")
        status_layout.addWidget(self.nba_data_status_label)
       
        self.position_stats_label = QLabel("")
        self.position_stats_label.setStyleSheet("color: #95a5a6; font-size: 10px; margin: 10px;")
        status_layout.addWidget(self.position_stats_label)
       
        layout.addWidget(status_group)
       
        # Update controls
        controls_group = QGroupBox("Data Management")
        controls_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
       
        controls_layout = QVBoxLayout(controls_group)
       
        # Update button
        self.update_nba_data_button = QPushButton("🔄 Update NBA Players Data")
        self.update_nba_data_button.setMinimumHeight(50)
        self.update_nba_data_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.update_nba_data_button.clicked.connect(self.update_nba_data)
        self.update_nba_data_button.setStyleSheet("""
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
        controls_layout.addWidget(self.update_nba_data_button)
       
        # Progress section
        self.nba_update_progress_label = QLabel("Ready to update")
        self.nba_update_progress_label.setAlignment(Qt.AlignCenter)
        self.nba_update_progress_label.setStyleSheet("color: white; font-weight: bold; margin: 10px;")
        controls_layout.addWidget(self.nba_update_progress_label)
       
        self.nba_update_progress_bar = QProgressBar()
        self.nba_update_progress_bar.setMinimumHeight(25)
        self.nba_update_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 3px;
            }
        """)
        controls_layout.addWidget(self.nba_update_progress_bar)
       
        layout.addWidget(controls_group)
       
        # Instructions
        instructions_text = QTextEdit()
        instructions_text.setMaximumHeight(200)
        instructions_text.setReadOnly(True)
        instructions_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 10px;
            }
        """)
        instructions_text.setPlainText("""
NBA Data Management Instructions:

🔄 Update NBA Players Data:
    • Fetches current roster data from NBA.com API
    • Automatically categorizes players by position (PG, SG, SF, PF, C)
    • Updates take 5-10 minutes depending on API response times
    • Recommended to update weekly during the season

📊 Data Usage:
    • Updated data is used for role-based training
    • Improves model accuracy by considering positional differences
    • Enables selection of players by position in training

⚠️  Notes:
    • Requires internet connection
    • NBA API rate limits apply
    • Data is cached locally for faster access
        """)
        layout.addWidget(instructions_text)
       
        layout.addStretch()
       
        self.tab_widget.addTab(nba_data_widget, "🏀 NBA Data")
   
    def create_status_bar(self):
        """Create the enhanced status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
       
        # Add permanent widgets to status bar
        self.model_status_label = QLabel("Model: Not Loaded")
        self.model_status_label.setStyleSheet("color: #e74c3c;")
       
        self.data_status_label = QLabel("Data: Ready")
        self.data_status_label.setStyleSheet("color: #27ae60;")
       
        self.nba_data_status_label_bar = QLabel("NBA Data: Checking...")
        self.nba_data_status_label_bar.setStyleSheet("color: #f39c12;")
       
        self.status_bar.addWidget(self.model_status_label)
        self.status_bar.addWidget(self.nba_data_status_label_bar)
        self.status_bar.addPermanentWidget(self.data_status_label)
       
        self.update_status("Ready - Enhanced with NBA data integration")
   
    # EVENT HANDLERS AND CORE METHODS
    def on_player_search_selected(self, player_name: str):
        """Handle player selection from search widget."""
        self.selected_player_name = player_name
       
        # Update the dropdown to match the search selection
        index = self.player_combo.findText(player_name)
        if index >= 0:
            self.player_combo.setCurrentIndex(index)
       
        # Enable predict button if model is loaded
        if self.is_model_loaded:
            self.predict_button.setEnabled(True)
   
    def on_dropdown_selection_changed(self, player_name: str):
        """Handle player selection from dropdown."""
        if player_name and player_name not in ["Load model first...", "No players found"]:
            self.selected_player_name = player_name
           
            # Update search widget to match dropdown selection
            self.player_search_widget.set_selected_player(player_name)
           
            # Enable predict button if model is loaded
            if self.is_model_loaded:
                self.predict_button.setEnabled(True)
   
    def on_training_method_changed(self):
        """Handle training method selection change with role support."""
        method = self.training_method_combo.currentText()
       
        if method == "Role-Based Training (Recommended)":
            self.role_selection_widget.setVisible(True)
            self.training_players_entry.setVisible(False)
        elif method == "Enter Custom Players":
            self.role_selection_widget.setVisible(False)
            self.training_players_entry.setVisible(True)
        else:  # Train All Available Players
            self.role_selection_widget.setVisible(False)
            self.training_players_entry.setVisible(False)
   
    def update_players_per_role_label(self, value):
        """Update players per role label."""
        self.players_per_role_label.setText(f"{value} players")
   
    def update_recent_games_label(self, value):
        """Update recent games label."""
        self.recent_games_label.setText(f"{value} games")
   
    def show_training_tab(self):
        """Switch to training tab."""
        self.tab_widget.setCurrentIndex(2)
   
    # NBA DATA MANAGEMENT METHODS
    def ensure_nba_data_available(self):
        """Ensure NBA data is available for player search."""
        try:
            # Check if we have NBA data
            nba_data = self.player_fetcher.load_players_data()
           
            if not nba_data:
                logger.info("No NBA data found, fetching automatically...")
               
                # Show a progress dialog
                from PyQt5.QtWidgets import QProgressDialog
                progress = QProgressDialog("Loading NBA players data...", "Cancel", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                QApplication.processEvents()
               
                try:
                    # Fetch NBA data
                    nba_data = self.player_fetcher.fetch_all_active_players()
                    progress.setValue(100)
                    progress.close()
                   
                    logger.info(f"Successfully loaded {nba_data['metadata']['total_players']} NBA players")
                   
                    # Refresh the player lists
                    self.refresh_players()
                    self.check_nba_data_status()
                   
                except Exception as e:
                    progress.close()
                    logger.error(f"Failed to fetch NBA data: {e}")
                    # Continue with fallback players
                   
        except Exception as e:
            logger.error(f"Error ensuring NBA data: {e}")
   
    def check_nba_data_status(self):
        """Check the status of NBA players data."""
        try:
            data = self.player_fetcher.load_players_data()
           
            if data:
                total_players = data['metadata']['total_players']
                last_updated = data['metadata']['last_updated'][:10]
                season = data['metadata']['season']
               
                # Update main status
                self.nba_data_status_label.setText(
                    f"✅ NBA Data Loaded: {total_players} players from {season} season (Updated: {last_updated})"
                )
                self.nba_data_status_label.setStyleSheet("color: #27ae60; font-size: 12px; margin: 10px;")
               
                # Update status bar
                self.nba_data_status_label_bar.setText(f"NBA Data: {total_players} players ({season})")
                self.nba_data_status_label_bar.setStyleSheet("color: #27ae60;")
               
                # Update position stats
                position_stats = self.player_fetcher.get_position_distribution()
                stats_text = " | ".join([f"{pos}: {count}" for pos, count in position_stats.items() if count > 0])
                self.position_stats_label.setText(f"Position breakdown: {stats_text}")
               
            else:
                self.nba_data_status_label.setText("❌ No NBA data available - Click 'Update NBA Players Data' to fetch current rosters")
                self.nba_data_status_label.setStyleSheet("color: #e74c3c; font-size: 12px; margin: 10px;")
               
                self.nba_data_status_label_bar.setText("NBA Data: Not available")
                self.nba_data_status_label_bar.setStyleSheet("color: #e74c3c;")
               
                self.position_stats_label.setText("No position data available")
               
        except Exception as e:
            logger.error(f"Error checking NBA data status: {e}")
            self.nba_data_status_label.setText(f"⚠️ Error checking NBA data: {str(e)}")
            self.nba_data_status_label.setStyleSheet("color: #f39c12; font-size: 12px; margin: 10px;")
   
    def update_nba_data(self):
        """Update NBA players data with progress tracking."""
        if self.nba_update_worker and self.nba_update_worker.isRunning():
            QMessageBox.warning(self, "Warning", "NBA data update is already in progress.")
            return
       
        reply = QMessageBox.question(
            self, "Update NBA Players Data",
            "This will fetch current NBA roster data from the NBA API.\n"
            "The process may take 5-10 minutes depending on API response times.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
       
        if reply == QMessageBox.Yes:
            # Update UI state
            self.update_nba_data_button.setEnabled(False)
            self.update_nba_data_button.setText("Updating...")
            self.nba_update_progress_bar.setValue(0)
           
            # Create and start worker
            self.nba_update_worker = NBADataUpdateWorker()
           
            # Connect signals
            self.nba_update_worker.progress_updated.connect(self.nba_update_progress_bar.setValue)
            self.nba_update_worker.status_updated.connect(self.nba_update_progress_label.setText)
            self.nba_update_worker.update_completed.connect(self.on_nba_update_completed)
            self.nba_update_worker.update_failed.connect(self.on_nba_update_failed)
           
            # Start update
            self.nba_update_worker.start()
   
    def on_nba_update_completed(self, data):
        """Handle NBA data update completion."""
        # Reset UI
        self.update_nba_data_button.setEnabled(True)
        self.update_nba_data_button.setText("🔄 Update NBA Players Data")
       
        # Update status displays
        self.check_nba_data_status()
        self.refresh_players()
       
        # Clear role mapping cache
        self.player_roles._cached_role_mapping = None
       
        # Show success message
        total_players = data['metadata']['total_players']
        season = data['metadata']['season']
       
        QMessageBox.information(
            self, "Update Complete",
            f"Successfully updated NBA players data!\n\n"
            f"Total players: {total_players}\n"
            f"Season: {season}\n\n"
            f"Position breakdown:\n" +
            "\n".join([f"• {pos}: {len(players)} players" 
                        for pos, players in data['players_by_position'].items() 
                        if len(players) > 0])
        )
   
    def on_nba_update_failed(self, error_message):
        """Handle NBA data update failure."""
        # Reset UI
        self.update_nba_data_button.setEnabled(True)
        self.update_nba_data_button.setText("🔄 Update NBA Players Data")
       
        QMessageBox.critical(self, "Update Failed", f"Failed to update NBA players data:\n\n{error_message}")
   
    # MODEL MANAGEMENT METHODS
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
   
    # TRAINING METHODS
    def start_training(self):
        """Enhanced start training with role support."""
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.warning(self, "Warning", "Training is already in progress.")
            return
       
        # Get training parameters based on method
        method = self.training_method_combo.currentText()
        player_names = None
        roles = None
        role_based = False
       
        if method == "Role-Based Training (Recommended)":
            # Get selected roles
            selected_roles = [role for role, checkbox in self.role_checkboxes.items() if checkbox.isChecked()]
            if not selected_roles:
                QMessageBox.warning(self, "Warning", "Please select at least one position for role-based training.")
                return
           
            # Check if NBA data is available
            nba_data = self.player_fetcher.load_players_data()
            if not nba_data:
                reply = QMessageBox.question(
                    self, "NBA Data Required",
                    "Role-based training requires NBA player data.\n"
                    "Would you like to fetch it now? (This may take 5-10 minutes)",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.update_nba_data()
                    return
                else:
                    QMessageBox.warning(self, "Warning", "Cannot proceed with role-based training without NBA data.")
                    return
           
            roles = selected_roles
            role_based = True
           
        elif method == "Enter Custom Players":
            players_text = self.training_players_entry.text().strip()
            if not players_text:
                QMessageBox.warning(self, "Warning", "Please enter player names for training.")
                return
            player_names = [name.strip() for name in players_text.split(',')]
           
        else:  # Train All Available Players
            player_names = None
       
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
       
        # Store the training parameters
        self.current_training_players = player_names
        self.current_training_roles = roles
       
        # Update UI
        self.start_training_button.setEnabled(False)
        self.stop_training_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.training_log.clear()
       
        # Create and start worker
        self.training_worker = TrainingWorker(
            self.predictor, player_names, roles, optimize, use_cache, role_based
        )
       
        # Connect signals
        self.training_worker.progress_updated.connect(self.progress_bar.setValue)
        self.training_worker.status_updated.connect(self.progress_label.setText)
        self.training_worker.log_updated.connect(self.add_training_log)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.training_failed.connect(self.on_training_failed)
       
        # Start training
        self.training_worker.start()
       
        if role_based:
            self.update_status(f"Role-based training started for positions: {', '.join(roles)}")
        else:
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
        """Enhanced training completion handler."""
        self.is_model_loaded = True
        self.update_ui_state()
        self.refresh_players()
       
        # Add trained players to storage
        if hasattr(self, 'current_training_players') and self.current_training_players:
            from utils.player_storage import PlayerStorage
            storage = PlayerStorage()
            storage.add_trained_players(self.current_training_players)
       
        # Reset training UI
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
       
        self.update_status("Training completed successfully!")
       
        # Show results
        best_model = min(results.keys(), key=lambda k: results[k]['test_mae'])
        best_mae = results[best_model]['test_mae']
       
        training_type = "Role-based" if hasattr(self, 'current_training_roles') and self.current_training_roles else "Standard"
       
        QMessageBox.information(
            self, "Training Complete",
            f"{training_type} model training completed successfully!\n\n"
            f"Best Model: {best_model.title()}\n"
            f"Best MAE: {best_mae:.3f} points\n\n"
            f"The model is now ready for predictions."
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
   
    # PREDICTION METHODS
    def refresh_players(self):
        """Enhanced refresh players method with search widget support."""
        try:
            # Get players from multiple sources
            all_available_players = []
           
            # 1. Try to get trained players (if model is loaded)
            if self.is_model_loaded:
                try:
                    trained_players = self.predictor.get_available_players()
                    all_available_players.extend(trained_players)
                    logger.info(f"Found {len(trained_players)} trained players")
                except Exception as e:
                    logger.warning(f"Could not get trained players: {e}")
           
            # 2. Get players from NBA data (this is the key addition)
            try:
                nba_data = self.player_fetcher.load_players_data()
                if nba_data and 'all_players' in nba_data:
                    nba_player_names = [player['name'] for player in nba_data['all_players']]
                    all_available_players.extend(nba_player_names)
                    logger.info(f"Found {len(nba_player_names)} NBA players")
                else:
                    logger.info("No NBA data available, trying to fetch...")
                    # If no NBA data, try to get popular players as fallback
                    from utils.player_storage import PlayerStorage
                    storage = PlayerStorage()
                    popular_players = storage.get_popular_players()
                    all_available_players.extend(popular_players)
                    logger.info(f"Using {len(popular_players)} popular players as fallback")
            except Exception as e:
                logger.warning(f"Could not get NBA players: {e}")
           
            # 3. Remove duplicates while preserving order
            unique_players = []
            seen = set()
            for player in all_available_players:
                if player not in seen:
                    unique_players.append(player)
                    seen.add(player)
           
            # 4. Update the dropdown (traditional method)
            self.player_combo.clear()
            if unique_players:
                # Prioritize trained players at the top
                if self.is_model_loaded:
                    trained_players = []
                    other_players = []
                   
                    try:
                        cached_players = set(self.predictor.get_available_players())
                        for player in unique_players:
                            if player in cached_players:
                                trained_players.append(player)
                            else:
                                other_players.append(player)
                       
                        # Add trained players first, then others
                        self.player_combo.addItems(trained_players + other_players)
                    except:
                        self.player_combo.addItems(unique_players)
                else:
                    self.player_combo.addItems(unique_players)
               
                self.player_combo.setEnabled(True)
            else:
                self.player_combo.addItem("No players available")
                self.player_combo.setEnabled(False)
           
            # 5. Update search widget (this is the most important part)
            self.player_search_widget.update_player_list(unique_players)
           
            logger.info(f"Refreshed player list with {len(unique_players)} total players")
           
        except Exception as e:
            logger.error(f"Error refreshing players: {e}")
            self.player_combo.clear()
            self.player_combo.addItem("Error loading players")
            self.player_combo.setEnabled(False)
            self.player_search_widget.update_player_list([])
   
    def predict_player_points(self):
        """Enhanced predict player points method."""
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a trained model first.")
            return
       
        # Get player name from search widget or dropdown
        player_name = getattr(self, 'selected_player_name', None)
       
        if not player_name:
            # Fallback to dropdown selection
            player_name = self.player_combo.currentText()
       
        if not player_name or player_name in ["Load model first...", "No players found", "Error loading players"]:
            QMessageBox.warning(self, "Warning", "Please search for and select a valid player.")
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
        """Display enhanced prediction results with role context."""
        player_name = predictions.get('player_name', 'Unknown Player')
        recent_avg = predictions.get('recent_average', 0)
       
        # Try to get player role information
        player_role = "Unknown"
        try:
            nba_data = self.player_fetcher.load_players_data()
            if nba_data:
                for player in nba_data['all_players']:
                    if player['name'].lower() == player_name.lower():
                        player_role = player['position']
                        break
        except:
            pass
       
        # Build result text
        result_text = f"🏀 PREDICTION RESULTS FOR {player_name.upper()}\n"
        result_text += "=" * 70 + "\n\n"
       
        # Player context
        result_text += f"📊 PLAYER CONTEXT:\n"
        result_text += f"Position: {player_role}\n"
        result_text += f"Recent Average ({self.recent_games_slider.value()} games): {recent_avg:.1f} points\n\n"
       
        # Model predictions
        result_text += "🤖 MODEL PREDICTIONS:\n"
        result_text += "-" * 50 + "\n"
       
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
       
        # Enhanced analysis with role context
        ensemble_pred = predictions.get('ensemble', {}).get('predicted_points', 0)
        diff_from_avg = ensemble_pred - recent_avg
       
        result_text += "\n" + "=" * 70 + "\n"
        result_text += "📈 ENHANCED ANALYSIS:\n"
        result_text += "-" * 50 + "\n"
       
        # Role-based context
        if player_role != "Unknown":
            role_characteristics = self.player_roles.ROLE_CHARACTERISTICS.get(player_role, {})
            typical_range = role_characteristics.get('scoring_range', (10, 30))
           
            result_text += f"• Position Analysis ({player_role}):\n"
            result_text += f"  - Typical {player_role} scoring range: {typical_range[0]}-{typical_range[1]} points\n"
           
            if ensemble_pred < typical_range[0]:
                result_text += f"  - ⬇️ Below typical {player_role} scoring range\n"
            elif ensemble_pred > typical_range[1]:
                result_text += f"  - ⬆️ Above typical {player_role} scoring range\n"
            else:
                result_text += f"  - ✅ Within typical {player_role} scoring range\n"
       
        # Trend analysis
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
       
        # Enhanced Fantasy/Betting Insights
        result_text += "\n" + "🎯 FANTASY/BETTING INSIGHTS:\n"
        result_text += "-" * 50 + "\n"
       
        if confidence == "HIGH" and ensemble_pred > recent_avg + 2:
            result_text += "• 🔥 STRONG BUY: Model confident in over-performance\n"
        elif confidence == "HIGH" and ensemble_pred < recent_avg - 2:
            result_text += "• 🧊 FADE PLAY: Model confident in under-performance\n"
        elif confidence == "MEDIUM":
            result_text += "• ⚖️ NEUTRAL: Moderate confidence, proceed with caution\n"
        else:
            result_text += "• ❓ LOW CONFIDENCE: High variance expected\n"
       
        # Role-specific insights
        if player_role != "Unknown":
            result_text += f"• 🏀 {player_role} Specific: Consider matchup and game pace\n"
       
        # Model information
        result_text += "\n" + "ℹ️  MODEL INFORMATION:\n"
        result_text += "-" * 50 + "\n"
        result_text += "• Enhanced with role-based features\n"
        result_text += "• Trained on position-specific patterns\n"
        result_text += "• Uses ensemble of 5 machine learning models\n"
        result_text += "• Incorporates 100+ basketball analytics features\n"
       
        # Disclaimer
        result_text += "\n" + "⚠️  DISCLAIMER:\n"
        result_text += "Predictions based on historical performance and statistical models.\n"
        result_text += "Actual results may vary due to injuries, matchups, and other factors.\n"
        result_text += "Use for entertainment and analysis purposes only.\n"
       
        self.results_text.setPlainText(result_text)
   
    # ANALYSIS METHODS
    def show_model_performance(self):
        """Show model performance analysis."""
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Warning", "Please load or train a model first.")
            return
       
        try:
            performance_df = self.predictor.get_model_performance()
            # Fix column name reference
            performance_df = performance_df.rename(columns={'Test R-squared': 'Test R'})
            self.plot_widget.plot_model_performance(performance_df)
            self.tab_widget.setCurrentIndex(1)
           
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
   
    # UI STATE MANAGEMENT
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
   
    # WINDOW MANAGEMENT
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
        elif self.nba_update_worker and self.nba_update_worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "NBA data update is in progress. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No
            )
           
            if reply == QMessageBox.Yes:
                self.nba_update_worker.terminate()
                self.nba_update_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
   
    def create_team_comparison_tab(self):
        """Create the team comparison tab."""
        team_comp_widget = QWidget()
        layout = QVBoxLayout(team_comp_widget)
    
        # Header
        header_label = QLabel("🏀 Team vs Team Comparison")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("color: #3498db; margin: 10px;")
        layout.addWidget(header_label)
    
        # Team selection section
        selection_group = QGroupBox("Team Selection")
        selection_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
    
        selection_layout = QHBoxLayout(selection_group)
    
        # Complete list of all 30 NBA teams
        all_nba_teams = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
            "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
            "Utah Jazz", "Washington Wizards"
        ]
    
        # Team A selection
        team_a_layout = QVBoxLayout()
        team_a_label = QLabel("Team A:")
        team_a_label.setStyleSheet("color: white; font-weight: bold; margin-bottom: 5px;")
        team_a_layout.addWidget(team_a_label)
    
        self.team_a_combo = QComboBox()
        self.team_a_combo.addItems(all_nba_teams)
        self.team_a_combo.setCurrentText("Los Angeles Lakers")  # Default selection
        self.team_a_combo.setStyleSheet("""
            QComboBox {
                padding: 12px;
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #2c3e50;
                color: white;
                min-height: 25px;
                font-size: 12px;
                font-weight: bold;
            }
            QComboBox:hover {
                border: 2px solid #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid white;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
                selection-color: white;
                border: 1px solid #555;
                padding: 5px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px;
                border-bottom: 1px solid #555;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #34495e;
            }
        """)
        team_a_layout.addWidget(self.team_a_combo)
    
        # VS label with better styling
        vs_widget = QWidget()
        vs_layout = QVBoxLayout(vs_widget)
        vs_layout.setAlignment(Qt.AlignCenter)
    
        vs_label = QLabel("🆚")
        vs_label.setFont(QFont("Arial", 32, QFont.Bold))
        vs_label.setAlignment(Qt.AlignCenter)
        vs_label.setStyleSheet("color: #e74c3c; margin: 10px 30px; padding: 10px;")
    
        vs_text = QLabel("VS")
        vs_text.setFont(QFont("Arial", 12, QFont.Bold))
        vs_text.setAlignment(Qt.AlignCenter)
        vs_text.setStyleSheet("color: #e74c3c; margin: 0;")
    
        vs_layout.addWidget(vs_label)
        vs_layout.addWidget(vs_text)
    
        # Team B selection
        team_b_layout = QVBoxLayout()
        team_b_label = QLabel("Team B:")
        team_b_label.setStyleSheet("color: white; font-weight: bold; margin-bottom: 5px;")
        team_b_layout.addWidget(team_b_label)
    
        self.team_b_combo = QComboBox()
        self.team_b_combo.addItems(all_nba_teams)
        self.team_b_combo.setCurrentText("Golden State Warriors")  # Default to different team
        self.team_b_combo.setStyleSheet(self.team_a_combo.styleSheet())
        team_b_layout.addWidget(self.team_b_combo)
    
        # Add random matchup button
        random_matchup_layout = QVBoxLayout()
        random_matchup_layout.addStretch()
    
        self.random_matchup_button = QPushButton("🎲 Random\nMatchup")
        self.random_matchup_button.setMinimumSize(80, 60)
        self.random_matchup_button.clicked.connect(self.generate_random_matchup)
        self.random_matchup_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        random_matchup_layout.addWidget(self.random_matchup_button)
        random_matchup_layout.addStretch()
    
        # Combine team selection layouts
        selection_layout.addLayout(team_a_layout, 2)  # Give more space
        selection_layout.addWidget(vs_widget, 0)      # Minimal space for VS
        selection_layout.addLayout(team_b_layout, 2)  # Give more space
        selection_layout.addLayout(random_matchup_layout, 0)  # Minimal space for button
    
        layout.addWidget(selection_group)
    
        # Rest of the method stays the same...
        # Game context section
        context_group = QGroupBox("Game Context (Optional)")
        context_group.setStyleSheet(selection_group.styleSheet())
        context_layout = QGridLayout(context_group)
    
        # Home team selection
        home_label = QLabel("Home Team:")
        home_label.setStyleSheet("color: white; font-weight: bold;")
        context_layout.addWidget(home_label, 0, 0)
    
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItems(["Neutral Site", "Team A", "Team B"])
        self.home_team_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2c3e50;
                color: white;
                min-height: 20px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
            }
        """)
        context_layout.addWidget(self.home_team_combo, 0, 1)
    
        # Rest differential
        rest_label = QLabel("Rest Advantage:")
        rest_label.setStyleSheet("color: white; font-weight: bold;")
        context_layout.addWidget(rest_label, 0, 2)
    
        self.rest_combo = QComboBox()
        self.rest_combo.addItems([
            "Equal Rest", "Team A (+1 day)", "Team A (+2 days)",
            "Team B (+1 day)", "Team B (+2 days)"
        ])
        self.rest_combo.setStyleSheet(self.home_team_combo.styleSheet())
        context_layout.addWidget(self.rest_combo, 0, 3)
    
        layout.addWidget(context_group)
    
        # Compare button
        self.compare_teams_button = QPushButton("🔥 COMPARE TEAMS")
        self.compare_teams_button.setMinimumHeight(50)
        self.compare_teams_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.compare_teams_button.clicked.connect(self.compare_teams)
        self.compare_teams_button.setStyleSheet("""
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
        layout.addWidget(self.compare_teams_button)
    
        # Results area (same as before)
        results_group = QGroupBox("Comparison Results")
        results_group.setStyleSheet(selection_group.styleSheet())
        results_layout = QVBoxLayout(results_group)
    
        self.team_comparison_results = QTextEdit()
        self.team_comparison_results.setFont(QFont("Consolas", 10))
        self.team_comparison_results.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 10px;
            }
        """)
        self.team_comparison_results.setPlainText(
            "Select two teams and click 'COMPARE TEAMS' to see a detailed analysis...\n\n"
            "🏀 Team Comparison Features:\n"
            "• Individual player predictions for each team\n"
            "• Team total scoring projections\n"
            "• Win probability calculations\n"
            "• Depth analysis and bench strength\n"
            "• Positional matchup breakdowns\n"
            "• Game context adjustments (home court, rest)\n"
            "• Monte Carlo simulation results\n"
            "• Confidence intervals and uncertainty analysis\n\n"
            f"📊 Available Teams: {len(all_nba_teams)} NBA teams loaded"
        )
        results_layout.addWidget(self.team_comparison_results)
    
        layout.addWidget(results_group)
    
        self.tab_widget.addTab(team_comp_widget, "🏀 Team Comparison")

        # Add this helper method for the random matchup button:
    def generate_random_matchup(self):
        """Generate a random team matchup."""
        import random
    
        all_teams = [self.team_a_combo.itemText(i) for i in range(self.team_a_combo.count())]
    
        # Select two different random teams
        team_a = random.choice(all_teams)
        remaining_teams = [t for t in all_teams if t != team_a]
        team_b = random.choice(remaining_teams)
    
        self.team_a_combo.setCurrentText(team_a)
        self.team_b_combo.setCurrentText(team_b)
    
        # Also randomize game context for fun
        self.home_team_combo.setCurrentIndex(random.randint(0, 2))
        self.rest_combo.setCurrentIndex(random.randint(0, 4))
    
        self.update_status(f"Random matchup: {team_a} vs {team_b}")

    def compare_teams(self):
        """Enhanced team comparison method."""
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Warning", "Please load or train a model first.")
            return

        team_a = self.team_a_combo.currentText()
        team_b = self.team_b_combo.currentText()

        if team_a == team_b:
            QMessageBox.warning(self, "Warning", "Please select two different teams.")
            return

        try:
            self.update_status(f"Comparing teams: {team_a} vs {team_b}")
            self.team_comparison_results.setPlainText("Generating comprehensive team comparison...\n")
            QApplication.processEvents()
        
            # Get game context
            game_context = {
                'home_team': None,
                'rest_differential': 0
            }
        
            # Parse home team selection
            home_selection = self.home_team_combo.currentText()
            if home_selection == "Team A":
                game_context['home_team'] = 'team_a'
            elif home_selection == "Team B":
                game_context['home_team'] = 'team_b'
        
            # Parse rest differential
            rest_selection = self.rest_combo.currentText()
            if "Team A" in rest_selection:
                if "+1 day" in rest_selection:
                    game_context['rest_differential'] = 1
                elif "+2 days" in rest_selection:
                    game_context['rest_differential'] = 2
            elif "Team B" in rest_selection:
                if "+1 day" in rest_selection:
                    game_context['rest_differential'] = -1
                elif "+2 days" in rest_selection:
                    game_context['rest_differential'] = -2
        
            # Import and use enhanced team comparison
            try:
                from src.team_comparison import EnhancedTeamComparison
            except ImportError:
                QMessageBox.critical(
                    self, "Import Error", 
                    "Team comparison functionality is not available.\n"
                    "Please check the installation."
                )
                return
        
            # Use enhanced team comparison
            team_comparator = EnhancedTeamComparison(self.predictor)
        
            # Get comprehensive comparison
            comparison_results = team_comparator.compare_teams_comprehensive(
                team_a, team_b, game_context
            )
        
            # Display detailed results
            self.display_team_comparison_results(comparison_results)
            self.update_status("Team comparison completed successfully!")
        
        except Exception as e:
            error_msg = f"Error comparing teams: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.team_comparison_results.setPlainText(f"Error: {error_msg}")
            self.update_status("Team comparison failed")

    def display_team_comparison_results(self, results: Dict):
        """Display comprehensive team comparison results."""
        team_a = results['teams']['team_a']
        team_b = results['teams']['team_b']
    
        # Build comprehensive result text
        result_text = f"🏀 COMPREHENSIVE TEAM COMPARISON\n"
        result_text += "=" * 80 + "\n"
        result_text += f"🆚 {team_a.upper()} vs {team_b.upper()}\n"
        result_text += "=" * 80 + "\n\n"
    
        # Main Predictions
        ensemble = results['predictions']['ensemble']
        result_text += f"🎯 FINAL PREDICTION:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        result_text += f"🏆 WINNER: {team_a if ensemble['win_probability_a'] > 0.5 else team_b}\n"
        result_text += f"📊 Win Probability: {team_a} {ensemble['win_probability_a']:.1%} | {team_b} {ensemble['win_probability_b']:.1%}\n"
        result_text += f"🏀 Predicted Score: {team_a} {ensemble['team_a_score']} - {ensemble['team_b_score']} {team_b}\n"
        result_text += f"📈 Spread: {team_a} {ensemble['spread']:+.1f}\n"
        result_text += f"🎯 Total Points: {ensemble['total']}\n\n"
    
        # Confidence Analysis
        confidence = results['confidence_analysis']
        result_text += f"📊 PREDICTION CONFIDENCE:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        result_text += f"Overall Confidence: {confidence['confidence_grade']} ({confidence['overall_confidence']:.1%})\n"
        result_text += f"Model Uncertainty: ±{confidence['model_uncertainty']:.1f} points\n"
        result_text += f"Prediction Consistency: {confidence['prediction_consistency']:.1%}\n\n"
    
        # Uncertainty Factors
        result_text += f"⚠️  Key Uncertainty Factors:\n"
        for factor in confidence['uncertainty_factors']:
            result_text += f"   • {factor}\n"
        result_text += "\n"
    
        # Multiple Prediction Methods
        result_text += f"🤖 MULTIPLE PREDICTION METHODS:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
        predictions = results['predictions']
        method_names = {
            'direct_aggregation': 'Player Aggregation',
            'possession_based': 'Possession Model',
            'matchup_adjusted': 'Matchup Adjusted',
            'context_adjusted': 'Context Adjusted',
            'ensemble': 'Ensemble Average'
        }
    
        for method, pred_data in predictions.items():
            if method in method_names:
                method_name = method_names[method]
                score_a = pred_data['team_a_score']
                score_b = pred_data['team_b_score']
                win_prob = pred_data['win_probability_a']
            
                result_text += f"{method_name:18s}: {team_a} {score_a:5.1f} - {score_b:5.1f} {team_b} | Win%: {win_prob:.1%}\n"
    
        # Team Metrics Comparison
        team_a_metrics = results['team_metrics']['team_a']
        team_b_metrics = results['team_metrics']['team_b']
    
        result_text += f"\n📈 TEAM METRICS COMPARISON:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        result_text += f"{'Metric':<20s} | {team_a:<15s} | {team_b:<15s} | Advantage\n"
        result_text += f"{'-'*20} | {'-'*15} | {'-'*15} | {'-'*15}\n"
    
        metrics_to_show = [
            ('Total Predicted Pts', 'total_predicted_points'),
            ('Starter Strength', 'starter_strength'),
            ('Bench Strength', 'bench_strength'),
            ('Depth Score', 'depth_score'),
            ('Estimated Pace', 'estimated_pace'),
            ('Offensive Rating', 'offensive_rating'),
            ('Defensive Rating', 'defensive_rating')
        ]
    
        for metric_name, metric_key in metrics_to_show:
            val_a = team_a_metrics.get(metric_key, 0)
            val_b = team_b_metrics.get(metric_key, 0)
        
            if metric_key == 'defensive_rating':  # Lower is better for defense
                advantage = team_a if val_a < val_b else team_b if val_b < val_a else "Even"
            else:  # Higher is better for other metrics
                advantage = team_a if val_a > val_b else team_b if val_b > val_a else "Even"
        
            result_text += f"{metric_name:<20s} | {val_a:<15.1f} | {val_b:<15.1f} | {advantage}\n"
    
        # Matchup Analysis
        matchup = results['matchup_analysis']
        result_text += f"\n🥊 MATCHUP ANALYSIS:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
        # Positional matchups
        result_text += f"Position-by-Position Breakdown:\n"
        for pos, pos_data in matchup['positional_advantages'].items():
            advantage = pos_data['advantage']
            diff = pos_data['point_differential']
        
            if advantage == 'team_a':
                result_text += f"   {pos}: {team_a} advantage (+{diff:.1f} pts)\n"
            elif advantage == 'team_b':
                result_text += f"   {pos}: {team_b} advantage (+{abs(diff):.1f} pts)\n"
            else:
                result_text += f"   {pos}: Even matchup\n"
    
        # Pace matchup
        pace_info = matchup['pace_matchup']
        result_text += f"\nPace Analysis:\n"
        result_text += f"   Expected Game Pace: {pace_info['expected_game_pace']} possessions\n"
        result_text += f"   {pace_info['analysis']}\n"
    
        # Key factors
        result_text += f"\n🔑 KEY FACTORS:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for factor in matchup['key_factors']:
            result_text += f"   • {factor}\n"
    
        # Monte Carlo Simulation Results
        monte_carlo = results['monte_carlo']
        result_text += f"\n🎲 MONTE CARLO SIMULATION ({monte_carlo['simulations_run']:,} simulations):\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
        # Score distributions
        team_a_dist = monte_carlo['score_distributions']['team_a']
        team_b_dist = monte_carlo['score_distributions']['team_b']
    
        result_text += f"{team_a} Score Distribution:\n"
        result_text += f"   Mean: {team_a_dist['mean']} ± {team_a_dist['std']:.1f}\n"
        result_text += f"   Range: {team_a_dist['percentiles']['10th']} - {team_a_dist['percentiles']['90th']} (80% confidence)\n"
    
        result_text += f"\n{team_b} Score Distribution:\n"
        result_text += f"   Mean: {team_b_dist['mean']} ± {team_b_dist['std']:.1f}\n"
        result_text += f"   Range: {team_b_dist['percentiles']['10th']} - {team_b_dist['percentiles']['90th']} (80% confidence)\n"
    
        # Game outcome probabilities
        margin_analysis = monte_carlo['margin_analysis']
        result_text += f"\nGame Outcome Probabilities:\n"
        result_text += f"   Close Game (≤5 pts): {margin_analysis['close_game_probability']:.1%}\n"
        result_text += f"   {team_a} Blowout (≥15 pts): {margin_analysis['blowout_probability_a']:.1%}\n"
        result_text += f"   {team_b} Blowout (≥15 pts): {margin_analysis['blowout_probability_b']:.1%}\n"
    
        # Game Context (if provided)
        context = results.get('game_context', {})
        if context:
            result_text += f"\n🏟️ GAME CONTEXT ADJUSTMENTS:\n"
            result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
            if context.get('home_team'):
                home_team = team_a if context['home_team'] == 'team_a' else team_b
                result_text += f"   🏠 Home Court: {home_team} (+3 pts typical advantage)\n"
        
            rest_diff = context.get('rest_differential', 0)
            if rest_diff != 0:
                rested_team = team_a if rest_diff > 0 else team_b
                result_text += f"   😴 Rest Advantage: {rested_team} (+{abs(rest_diff)} days rest)\n"
    
        # Fantasy/Betting Insights
        result_text += f"\n💰 FANTASY/BETTING INSIGHTS:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
        win_prob_diff = abs(ensemble['win_probability_a'] - 0.5)
    
        if confidence['confidence_grade'] == 'High' and win_prob_diff > 0.15:
            favored_team = team_a if ensemble['win_probability_a'] > 0.5 else team_b
            result_text += f"   🔥 STRONG PICK: {favored_team} (High confidence, clear favorite)\n"
        elif confidence['confidence_grade'] == 'Low' or win_prob_diff < 0.05:
            result_text += f"   ⚠️  HIGH VARIANCE: Very close game, small factors could decide\n"
        else:
            result_text += f"   ⚖️  MODERATE CONFIDENCE: Slight edge to predicted winner\n"
    
        # Point total recommendation
        if ensemble['total'] > 220:
            result_text += f"   📈 HIGH SCORING: Over {ensemble['total']:.0f} points expected\n"
        elif ensemble['total'] < 200:
            result_text += f"   📉 LOW SCORING: Under {ensemble['total']:.0f} points expected\n"
        else:
            result_text += f"   ➡️  AVERAGE SCORING: Around {ensemble['total']:.0f} points expected\n"
    
        # Model methodology
        result_text += f"\nℹ️  METHODOLOGY:\n"
        result_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        result_text += f"   • Uses ensemble of 4 prediction methods for robust analysis\n"
        result_text += f"   • Individual player predictions aggregated to team level\n"
        result_text += f"   • Advanced metrics: pace, efficiency, positional advantages\n"
        result_text += f"   • Monte Carlo simulation for uncertainty quantification\n"
        result_text += f"   • Game context adjustments (home court, rest, etc.)\n"
    
        # Disclaimer
        result_text += f"\n⚠️  DISCLAIMER:\n"
        result_text += f"Predictions based on current roster data and player performance models.\n"
        result_text += f"Actual results may vary due to injuries, lineup changes, and other factors.\n"
        result_text += f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
        self.team_comparison_results.setPlainText(result_text)
          
def main():
    """Main function to run the enhanced PyQt5 GUI application."""
    app = QApplication(sys.argv)
   
    # Set application properties
    app.setApplicationName("NBA Player Scoring Predictor")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Basketball Analytics Pro")
   
    # Apply dark theme if available
    if DARK_STYLE_AVAILABLE:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    else:
        # Enhanced dark theme fallback
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
            QGroupBox {
                font-weight: bold;
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
   
    # Create and show main window
    window = NBAPlayerScoringGUI()
    window.show()
   
    # Center window on screen
    screen = app.primaryScreen().geometry()
    window.move(
        (screen.width() - window.width()) // 2,
        (screen.height() - window.height()) // 2
    )
   
    logger.info("NBA Player Scoring Predictor GUI v2.0 started")
   
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()