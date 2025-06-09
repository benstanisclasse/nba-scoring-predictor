# -*- coding: utf-8 -*-
"""
Professional PyQt5 GUI for NBA Player Scoring Predictor - Enhanced with Live NBA Data
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
    """Enhanced Professional PyQt5 GUI application with category training and team predictions."""
    
    def __init__(self):
        super().__init__()
        self.predictor = EnhancedNBAPredictor()
        self.is_model_loaded = False
        self.training_worker = None
        self.nba_update_worker = None
        
        # Initialize NBA data components
        self.player_fetcher = NBAPlayerFetcher()
        self.player_roles = PlayerRoles()
        
        self.init_ui()
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_periodically)
        self.status_timer.start(5000)
    
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
        self.create_nba_data_tab()
        
        layout.addWidget(self.tab_widget)
    
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
        
        # Player selection (for specific player training)
        config_layout.addWidget(QLabel("Select Players:"), 2, 0)
        
        self.player_selection_widget = QWidget()
        player_selection_layout = QVBoxLayout(self.player_selection_widget)
        player_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        self.available_players_combo = QComboBox()
        self.available_players_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2c3e50;
                color: white;
                min-height: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
            }
        """)
        self.refresh_available_players()
        player_selection_layout.addWidget(self.available_players_combo)
        
        # Player management buttons
        player_buttons_layout = QHBoxLayout()
        
        self.add_player_button = QPushButton("➕ Add Selected")
        self.add_player_button.clicked.connect(self.add_selected_player)
        self.add_player_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        
        self.clear_players_button = QPushButton("🗑️ Clear All")
        self.clear_players_button.clicked.connect(self.clear_selected_players)
        self.clear_players_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        self.refresh_players_button = QPushButton("🔄 Refresh")
        self.refresh_players_button.clicked.connect(self.refresh_available_players)
        self.refresh_players_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        player_buttons_layout.addWidget(self.add_player_button)
        player_buttons_layout.addWidget(self.clear_players_button)
        player_buttons_layout.addWidget(self.refresh_players_button)
        player_buttons_layout.addStretch()
        
        player_selection_layout.addLayout(player_buttons_layout)
        
        # Selected players display
        self.selected_players_label = QLabel("Selected Players: None")
        self.selected_players_label.setStyleSheet("color: #3498db; font-weight: bold;")
        self.selected_players_label.setWordWrap(True)
        player_selection_layout.addWidget(self.selected_players_label)
        
        config_layout.addWidget(self.player_selection_widget, 2, 1)
        
        # Custom player entry
        config_layout.addWidget(QLabel("Custom Players:"), 3, 0)
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
        config_layout.addWidget(self.training_players_entry, 3, 1)
       
        # Training options
        self.optimize_checkbox = QCheckBox("Optimize hyperparameters (slower but better)")
        self.optimize_checkbox.setStyleSheet("color: white;")
        self.use_cache_checkbox = QCheckBox("Use cached data when available")
        self.use_cache_checkbox.setChecked(True)
        self.use_cache_checkbox.setStyleSheet("color: white;")
       
        config_layout.addWidget(self.optimize_checkbox, 4, 0, 1, 2)
        config_layout.addWidget(self.use_cache_checkbox, 5, 0, 1, 2)
       
        layout.addWidget(config_group)
       
        # Initialize selected players list
        self.selected_players = []
       
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
   
    def create_prediction_tab(self):
        """Create the prediction tab (unchanged from original)."""
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
        """Create the analysis tab (unchanged from original)."""
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
   
    # Enhanced methods for NBA data integration
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
        self.refresh_available_players()
       
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
   
    # Enhanced training methods
    def on_training_method_changed(self):
        """Handle training method selection change with role support."""
        method = self.training_method_combo.currentText()
       
        if method == "Role-Based Training (Recommended)":
            self.role_selection_widget.setVisible(True)
            self.player_selection_widget.setVisible(False)
            self.training_players_entry.setVisible(False)
        elif method == "Select Specific Players":
            self.role_selection_widget.setVisible(False)
            self.player_selection_widget.setVisible(True)
            self.training_players_entry.setVisible(False)
        elif method == "Enter Custom Players":
            self.role_selection_widget.setVisible(False)
            self.player_selection_widget.setVisible(False)
            self.training_players_entry.setVisible(True)
        else:  # Train All Available Players
            self.role_selection_widget.setVisible(False)
            self.player_selection_widget.setVisible(False)
            self.training_players_entry.setVisible(False)
   
    def refresh_available_players(self):
        """Refresh the available players dropdown with NBA data."""
        try:
            self.available_players_combo.clear()
           
            # Try to get players from NBA data first
            nba_data = self.player_fetcher.load_players_data()
           
            if nba_data:
                # Add players by position
                all_players = nba_data['all_players']
               
                self.available_players_combo.addItem("--- By Position ---")
               
                for position in ['PG', 'SG', 'SF', 'PF', 'C']:
                    position_players = [p for p in all_players if p['position'] == position]
                    if position_players:
                        self.available_players_combo.addItem(f"--- {position} ({len(position_players)} players) ---")
                        for player in sorted(position_players, key=lambda x: x['name'])[:10]:  # Limit to top 10
                            self.available_players_combo.addItem(f"  {player['name']} ({player['team_abbrev']})")
               
                # Add recent/popular players
                from utils.player_storage import PlayerStorage
                storage = PlayerStorage()
                trained_players = set(storage.get_trained_players())
               
                if trained_players:
                    self.available_players_combo.addItem("--- Previously Trained ---")
                    for player in sorted(trained_players):
                        self.available_players_combo.addItem(f"✓ {player}")
           
            else:
                # Fallback to original method
                from utils.player_storage import PlayerStorage
                storage = PlayerStorage()
               
                trained_players = storage.get_trained_players()
                popular_players = storage.get_popular_players()
               
                if trained_players:
                    self.available_players_combo.addItem("--- Previously Trained ---")
                    for player in trained_players:
                        self.available_players_combo.addItem(f"✓ {player}")
               
                self.available_players_combo.addItem("--- Popular Players ---")
                for player in popular_players:
                    if player not in trained_players:
                        self.available_players_combo.addItem(f"⭐ {player}")
           
            if self.available_players_combo.count() > 1:
                self.available_players_combo.setCurrentIndex(1)
               
        except Exception as e:
            logger.error(f"Error refreshing players: {e}")
            self.available_players_combo.addItem("Error loading players")
   
    def update_players_per_role_label(self, value):
        """Update players per role label."""
        self.players_per_role_label.setText(f"{value} players")
   
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
           
        elif method == "Select Specific Players":
            if not self.selected_players:
                QMessageBox.warning(self, "Warning", "Please select at least one player for training.")
                return
            player_names = self.selected_players.copy()
           
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
            self.refresh_available_players()
       
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
   
    # Keep all existing methods from the original GUI
    def add_selected_player(self):
        """Add the selected player to the training list."""
        current_text = self.available_players_combo.currentText()
       
        # Skip section headers
        if current_text.startswith("---"):
            return
       
        # Extract player name (remove prefix symbols and team info)
        player_name = current_text.replace("✓ ", "").replace("⭐ ", "").replace("  ", "")
        if "(" in player_name:
            player_name = player_name.split("(")[0].strip()
       
        if player_name and player_name not in self.selected_players:
           self.selected_players.append(player_name)
           self.update_selected_players_display()

    def clear_selected_players(self):
        """Clear all selected players."""
        self.selected_players = []
        self.update_selected_players_display()

    def update_selected_players_display(self):
        """Update the selected players display."""
        if self.selected_players:
            display_text = f"Selected Players ({len(self.selected_players)}): " + ", ".join(self.selected_players)
        else:
            display_text = "Selected Players: None"
       
        self.selected_players_label.setText(display_text)

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
        """Refresh the player dropdown list for predictions."""
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
    # Add these methods to your existing GUI class in src/gui.py

def create_enhanced_training_tab(self):
    """Create enhanced training tab with category support."""
    training_widget = QWidget()
    layout = QVBoxLayout(training_widget)
    
    # Training configuration
    config_group = QGroupBox("Enhanced Training Configuration")
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
    
    # Category selection
    config_layout.addWidget(QLabel("Training Category:"), 0, 0)
    
    self.category_combo = QComboBox()
    self.category_combo.addItems([
        "All Players",
        "Point Guards (PG)", 
        "Shooting Guards (SG)",
        "Small Forwards (SF)",
        "Power Forwards (PF)",
        "Centers (C)",
        "All Guards",
        "All Forwards", 
        "All Bigs (PF + C)",
        "Custom Player List"
    ])
    self.category_combo.setStyleSheet("""
        QComboBox {
            padding: 8px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #2c3e50;
            color: white;
            min-height: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: #2c3e50;
            color: white;
            selection-background-color: #3498db;
        }
    """)
    config_layout.addWidget(self.category_combo, 0, 1)
    
    # Max players per position
    config_layout.addWidget(QLabel("Max Players (0 = All):"), 1, 0)
    
    self.max_players_spinbox = QSpinBox()
    self.max_players_spinbox.setMinimum(0)
    self.max_players_spinbox.setMaximum(100)
    self.max_players_spinbox.setValue(0)
    self.max_players_spinbox.setStyleSheet("""
        QSpinBox {
            padding: 8px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #2c3e50;
            color: white;
        }
    """)
    config_layout.addWidget(self.max_players_spinbox, 1, 1)
    
    # Training options
    self.optimize_checkbox = QCheckBox("Optimize hyperparameters (slower but better)")
    self.optimize_checkbox.setStyleSheet("color: white;")
    self.use_cache_checkbox = QCheckBox("Use cached data when available")
    self.use_cache_checkbox.setChecked(True)
    self.use_cache_checkbox.setStyleSheet("color: white;")
    
    config_layout.addWidget(self.optimize_checkbox, 2, 0, 1, 2)
    config_layout.addWidget(self.use_cache_checkbox, 3, 0, 1, 2)
    
    layout.addWidget(config_group)
    
    # Training control
    control_layout = QHBoxLayout()
    
    self.start_category_training_button = QPushButton("🚀 START CATEGORY TRAINING")
    self.start_category_training_button.setMinimumHeight(50)
    self.start_category_training_button.setFont(QFont("Arial", 12, QFont.Bold))
    self.start_category_training_button.setStyleSheet("""
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
    self.start_category_training_button.clicked.connect(self.start_category_training)
   
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
   
    control_layout.addWidget(self.start_category_training_button)
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
   
    self.tab_widget.addTab(training_widget, "🏋️ Category Training")

def create_team_prediction_tab(self):
    """Create team vs team prediction tab."""
    team_widget = QWidget()
    layout = QVBoxLayout(team_widget)
   
    # Header
    header_label = QLabel("🏀 Team vs Team Prediction")
    header_label.setFont(QFont("Arial", 16, QFont.Bold))
    header_label.setAlignment(Qt.AlignCenter)
    header_label.setStyleSheet("color: #3498db; margin: 10px;")
    layout.addWidget(header_label)
   
    # Team selection
    teams_group = QGroupBox("Select Teams")
    teams_group.setStyleSheet("""
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
   
    teams_layout = QGridLayout(teams_group)
   
    # Team A selection
    teams_layout.addWidget(QLabel("Team A:"), 0, 0)
    self.team_a_combo = QComboBox()
    self.team_a_combo.addItems([
        "Los Angeles Lakers", "Golden State Warriors", "Boston Celtics",
        "Miami Heat", "Milwaukee Bucks", "Phoenix Suns", "Dallas Mavericks",
        "Denver Nuggets", "Philadelphia 76ers", "Brooklyn Nets",
        "Chicago Bulls", "New York Knicks", "Toronto Raptors"
    ])
    self.team_a_combo.setStyleSheet("""
        QComboBox {
            padding: 8px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #2c3e50;
            color: white;
            min-height: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: #2c3e50;
            color: white;
            selection-background-color: #3498db;
        }
    """)
    teams_layout.addWidget(self.team_a_combo, 0, 1)
   
    # Team B selection
    teams_layout.addWidget(QLabel("Team B:"), 1, 0)
    self.team_b_combo = QComboBox()
    self.team_b_combo.addItems([
        "Los Angeles Lakers", "Golden State Warriors", "Boston Celtics",
        "Miami Heat", "Milwaukee Bucks", "Phoenix Suns", "Dallas Mavericks",
        "Denver Nuggets", "Philadelphia 76ers", "Brooklyn Nets",
        "Chicago Bulls", "New York Knicks", "Toronto Raptors"
    ])
    self.team_b_combo.setCurrentIndex(1)  # Default to different team
    self.team_b_combo.setStyleSheet("""
        QComboBox {
            padding: 8px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #2c3e50;
            color: white;
            min-height: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: #2c3e50;
            color: white;
            selection-background-color: #3498db;
        }
    """)
    teams_layout.addWidget(self.team_b_combo, 1, 1)
   
    # Home team selection
    teams_layout.addWidget(QLabel("Home Team:"), 2, 0)
    self.home_team_combo = QComboBox()
    self.home_team_combo.addItems(["Team A", "Team B", "Neutral"])
    self.home_team_combo.setStyleSheet("""
        QComboBox {
            padding: 8px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #2c3e50;
            color: white;
            min-height: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: #2c3e50;
            color: white;
            selection-background-color: #3498db;
        }
    """)
    teams_layout.addWidget(self.home_team_combo, 2, 1)
   
    layout.addWidget(teams_group)
   
    # Prediction button
    self.predict_game_button = QPushButton("🎯 PREDICT GAME OUTCOME")
    self.predict_game_button.setMinimumHeight(50)
    self.predict_game_button.setFont(QFont("Arial", 12, QFont.Bold))
    self.predict_game_button.setStyleSheet("""
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
    self.predict_game_button.clicked.connect(self.predict_team_game)
    self.predict_game_button.setEnabled(False)  # Enable when model is loaded
   
    layout.addWidget(self.predict_game_button)
   
    # Results display
    results_group = QGroupBox("Game Prediction Results")
    results_group.setStyleSheet("""
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
   
    results_layout = QVBoxLayout(results_group)
   
    self.team_results_text = QTextEdit()
    self.team_results_text.setFont(QFont("Consolas", 10))
    self.team_results_text.setStyleSheet("""
        QTextEdit {
            background-color: #2c3e50;
            color: white;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 10px;
        }
    """)
    self.team_results_text.setPlainText("Load a trained model and select teams to see game predictions...")
    results_layout.addWidget(self.team_results_text)
   
    layout.addWidget(results_group)
   
    self.tab_widget.addTab(team_widget, "⚔️ Team vs Team")

def start_category_training(self):
    """Start category-based training."""
    if self.training_worker and self.training_worker.isRunning():
        QMessageBox.warning(self, "Warning", "Training is already in progress.")
        return
   
    # Get training parameters
    category_text = self.category_combo.currentText()
    max_players = self.max_players_spinbox.value() if self.max_players_spinbox.value() > 0 else None
    optimize = self.optimize_checkbox.isChecked()
    use_cache = self.use_cache_checkbox.isChecked()
   
    # Map GUI text to category codes
    category_mapping = {
        "All Players": "All",
        "Point Guards (PG)": "PG",
        "Shooting Guards (SG)": "SG", 
        "Small Forwards (SF)": "SF",
        "Power Forwards (PF)": "PF",
        "Centers (C)": "C",
        "All Guards": "Guards",
        "All Forwards": "Forwards",
        "All Bigs (PF + C)": "Bigs",
        "Custom Player List": "Custom"
    }
   
    category = category_mapping.get(category_text, "All")
   
    if category == "Custom":
        QMessageBox.information(self, "Not Implemented", 
                                "Custom player list training will be available in the next update.")
        return
   
    # Confirm training start
    if optimize:
        reply = QMessageBox.question(
            self, "Confirm Training",
            f"Starting {category} training with hyperparameter optimization.\n"
            f"This may take 30+ minutes.\n\n"
            f"Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
   
    # Update UI
    self.start_category_training_button.setEnabled(False)
    self.stop_training_button.setEnabled(True)
    self.progress_bar.setValue(0)
    self.training_log.clear()
   
    # Create and start worker
    self.training_worker = CategoryTrainingWorker(
        self.predictor, category, max_players, optimize, use_cache
    )
   
    # Connect signals
    self.training_worker.progress_updated.connect(self.progress_bar.setValue)
    self.training_worker.status_updated.connect(self.progress_label.setText)
    self.training_worker.log_updated.connect(self.add_training_log)
    self.training_worker.training_completed.connect(self.on_category_training_completed)
    self.training_worker.training_failed.connect(self.on_training_failed)
   
    # Start training
    self.training_worker.start()
   
    self.update_status(f"Category training started for: {category}")

# In GUI or other places where team prediction is used:
def predict_team_game(self):
    """Predict team vs team game outcome."""
    if not self.is_model_loaded:
        QMessageBox.warning(self, "Warning", "Please load or train a model first.")
        return
    
    team_a = self.team_a_combo.currentText()
    team_b = self.team_b_combo.currentText()
    
    if team_a == team_b:
        QMessageBox.warning(self, "Warning", "Please select different teams.")
        return
    
    # Get home team context
    home_selection = self.home_team_combo.currentText()
    game_context = {}
    
    if home_selection == "Team A":
        game_context['home_team'] = 'team_a'
    elif home_selection == "Team B":
        game_context['home_team'] = 'team_b'
    
    try:
        self.update_status(f"Predicting game: {team_a} vs {team_b}")
        self.team_results_text.setPlainText("Generating team predictions...\n")
        QApplication.processEvents()
        
        # Use get_team_predictor() instead of direct access
        team_predictor = self.predictor.get_team_predictor()
        prediction = team_predictor.predict_game(team_a, team_b, game_context)
        
        # Display results
        self.display_team_prediction_results(prediction)
        self.update_status("Team prediction completed successfully!")
        
    except Exception as e:
        error_msg = f"Error predicting game {team_a} vs {team_b}: {str(e)}"
        logger.error(error_msg)
        self.team_results_text.setPlainText(f"Error: {error_msg}")
        self.update_status("Team prediction failed")

def display_team_prediction_results(self, prediction: Dict):
    """Display team vs team prediction results."""
   
    team_a = prediction['team_a']
    team_b = prediction['team_b']
   
    # Build result text
    result_text = f"🏀 GAME PREDICTION: {team_a.upper()} vs {team_b.upper()}\n"
    result_text += "=" * 70 + "\n\n"
   
    # Win probability
    win_prob_a = prediction['winner_probability']['team_a']
    win_prob_b = prediction['winner_probability']['team_b']
   
    result_text += "🏆 WIN PROBABILITY:\n"
    result_text += f"{team_a}: {win_prob_a:.1%}\n"
    result_text += f"{team_b}: {win_prob_b:.1%}\n\n"
   
    # Predicted scores
    score_a = prediction['predicted_score']['team_a']
    score_b = prediction['predicted_score']['team_b']
   
    result_text += "📊 PREDICTED FINAL SCORE:\n"
    result_text += f"{team_a}: {score_a}\n"
    result_text += f"{team_b}: {score_b}\n\n"
   
    # Spread and totals
    spread = prediction['spread']
    total = prediction['total_points']
   
    result_text += "💰 BETTING INFORMATION:\n"
    if spread > 0:
        result_text += f"Spread: {team_a} -{abs(spread):.1f}\n"
    else:
        result_text += f"Spread: {team_b} -{abs(spread):.1f}\n"
   
    result_text += f"Total Points (O/U): {total:.1f}\n\n"
   
    # Key factors
    result_text += "🔑 KEY FACTORS:\n"
    for factor in prediction['key_factors']:
        result_text += f"• {factor}\n"
   
    result_text += "\n"
   
    # Team breakdowns if available
    if 'team_breakdowns' in prediction:
        result_text += "👥 TEAM BREAKDOWNS:\n"
        result_text += "-" * 50 + "\n"
       
        for team_key, team_name in [('team_a', team_a), ('team_b', team_b)]:
            if team_key in prediction['team_breakdowns']:
                result_text += f"\n{team_name}:\n"
                breakdown = prediction['team_breakdowns'][team_key]
               
                # Sort players by predicted points
                sorted_players = sorted(breakdown.items(), 
                                        key=lambda x: x[1]['predicted_points'], 
                                        reverse=True)
               
                for player, stats in sorted_players[:5]:  # Top 5 players
                    points = stats['predicted_points']
                    position = stats.get('position', 'N/A')
                    result_text += f"  {player} ({position}): {points:.1f} pts\n"
   
    # Confidence and disclaimer
    confidence = prediction.get('confidence', 0.5)
    result_text += f"\n📈 PREDICTION CONFIDENCE: {confidence:.1%}\n"
   
    result_text += "\n" + "⚠️  DISCLAIMER:\n"
    result_text += "Predictions based on individual player models and team composition.\n"
    result_text += "Actual results may vary due to injuries, coaching decisions, and game flow.\n"
    result_text += "Use for entertainment and analysis purposes only.\n"
   
    self.team_results_text.setPlainText(result_text)

def on_category_training_completed(self, results):
    """Handle category training completion."""
    self.is_model_loaded = True
    self.update_ui_state()
    self.refresh_players()
   
    # Enable team prediction
    self.predict_game_button.setEnabled(True)
   
    # Reset training UI
    self.start_category_training_button.setEnabled(True)
    self.stop_training_button.setEnabled(False)
   
    self.update_status("Category training completed successfully!")
   
    # Show results
    best_model = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_mae = results[best_model]['test_mae']
   
    QMessageBox.information(
        self, "Training Complete",
        f"Category model training completed successfully!\n\n"
        f"Best Model: {best_model.title()}\n"
        f"Best MAE: {best_mae:.3f} points\n\n"
        f"The model is now ready for individual and team predictions."
    )

def _simple_team_prediction(self, team_a: str, team_b: str, context: Dict) -> Dict:
    """Simple fallback team prediction when team predictor unavailable."""
   
    # This is a simplified version for when the full team predictor isn't available
    # You could enhance this by aggregating individual player predictions
   
    import random
   
    # Simulate team strength (in a real implementation, this would use actual data)
    team_a_strength = random.uniform(105, 120)
    team_b_strength = random.uniform(105, 120)
   
    # Apply home court advantage
    if context.get('home_team') == 'team_a':
        team_a_strength += 3
    elif context.get('home_team') == 'team_b':
        team_b_strength += 3
   
    # Calculate win probability
    diff = team_a_strength - team_b_strength
    win_prob_a = 1 / (1 + np.exp(-diff / 5))
   
    return {
        'team_a': team_a,
        'team_b': team_b,
        'winner_probability': {
            'team_a': win_prob_a,
            'team_b': 1 - win_prob_a
        },
        'predicted_score': {
            'team_a': team_a_strength,
            'team_b': team_b_strength
        },
        'spread': diff,
        'total_points': team_a_strength + team_b_strength,
        'confidence': 0.6,
        'key_factors': [
            f"Predicted as close matchup with {abs(diff):.1f} point differential",
            "Prediction based on simplified team model"
        ]
    }


# Category Training Worker Thread
class CategoryTrainingWorker(QThread):
    """Worker thread for category-based model training."""
   
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    training_failed = pyqtSignal(str)
   
    def __init__(self, predictor, category, max_players, optimize, use_cache):
        super().__init__()
        self.predictor = predictor
        self.category = category
        self.max_players = max_players
        self.optimize = optimize
        self.use_cache = use_cache
   
    def run(self):
        """Run category training in background thread."""
        try:
            self.status_updated.emit(f"Starting {self.category} training...")
            self.progress_updated.emit(10)
            self.log_updated.emit(f"Training category: {self.category}")
           
            if self.max_players:
                self.log_updated.emit(f"Max players per position: {self.max_players}")
           
            # Use the enhanced predictor's category training method
            results = self.predictor.train_by_category(
                category=self.category,
                max_players_per_position=self.max_players,
                optimize=self.optimize,
                use_cache=self.use_cache
            )
           
            self.progress_updated.emit(100)
            self.status_updated.emit("Training completed!")
           
            # Log results
            for model_name, metrics in results.items():
                self.log_updated.emit(
                    f"{model_name}: MAE={metrics['test_mae']:.3f}, R²={metrics['test_r2']:.3f}"
                )
           
            self.training_completed.emit(results)
           
        except Exception as e:
            error_msg = f"Category training failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.training_failed.emit(error_msg)

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