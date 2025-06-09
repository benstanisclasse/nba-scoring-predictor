"""
Custom widgets for NBA Player Scoring Predictor GUI
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import pandas as pd

class ModelComparisonTable(QWidget):
    """Custom widget for displaying model performance comparison."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Model Performance Comparison")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #3498db; margin: 10px;")
        layout.addWidget(header)
        
        # Table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #555;
                background-color: #404040;
                alternate-background-color: #4a4a4a;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.table)
    
    def update_data(self, performance_df: pd.DataFrame):
        """Update table with performance data."""
        self.table.setRowCount(len(performance_df))
        self.table.setColumnCount(len(performance_df.columns))
        self.table.setHorizontalHeaderLabels(performance_df.columns.tolist())
        
        for i, row in performance_df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                
                # Color coding for performance metrics
                if isinstance(value, (int, float)) and j > 0:  # Skip model name column
                    if j == 1:  # MAE column (lower is better)
                        if value < 5:
                            item.setBackground(QColor("#27ae60"))  # Green
                        elif value < 6:
                            item.setBackground(QColor("#f39c12"))  # Orange
                        else:
                            item.setBackground(QColor("#e74c3c"))  # Red
                    elif j == 3:  # R² column (higher is better)
                        if value > 0.7:
                            item.setBackground(QColor("#27ae60"))  # Green
                        elif value > 0.6:
                            item.setBackground(QColor("#f39c12"))  # Orange
                        else:
                            item.setBackground(QColor("#e74c3c"))  # Red
                
                self.table.setItem(i, j, item)
        
        # Adjust column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

class PredictionSummaryWidget(QWidget):
    """Widget for displaying prediction summary with visual indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Create summary cards
        self.create_summary_cards(layout)
    
    def create_summary_cards(self, layout):
        """Create summary cards for key metrics."""
        cards_layout = QHBoxLayout()
        
        # Ensemble Prediction Card
        self.ensemble_card = self.create_metric_card("Ensemble Prediction", "0.0", "#3498db")
        cards_layout.addWidget(self.ensemble_card)
        
        # Confidence Card
        self.confidence_card = self.create_metric_card("Confidence", "Medium", "#f39c12")
        cards_layout.addWidget(self.confidence_card)
        
        # Recent Average Card
        self.recent_avg_card = self.create_metric_card("Recent Average", "0.0", "#9b59b6")
        cards_layout.addWidget(self.recent_avg_card)
        
        # Trend Card
        self.trend_card = self.create_metric_card("Trend", "Neutral", "#95a5a6")
        cards_layout.addWidget(self.trend_card)
        
        layout.addLayout(cards_layout)
    
    def create_metric_card(self, title, value, color):
        """Create a metric card widget."""
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: white;")
        
        value_label = QLabel(str(value))
        value_label.setFont(QFont("Arial", 16, QFont.Bold))
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("color: white;")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        # Store reference to value label for updates
        card.value_label = value_label
        
        return card
    
    def update_predictions(self, predictions: dict):
        """Update the summary cards with new predictions."""
        ensemble_pred = predictions.get('ensemble', {}).get('predicted_points', 0)
        recent_avg = predictions.get('recent_average', 0)
        
        # Update ensemble prediction
        self.ensemble_card.value_label.setText(f"{ensemble_pred:.1f}")
        
        # Update confidence
        ensemble_mae = predictions.get('ensemble', {}).get('model_mae', 5)
        if ensemble_mae < 4:
            confidence = "HIGH"
            confidence_color = "#27ae60"
        elif ensemble_mae < 6:
            confidence = "MEDIUM" 
            confidence_color = "#f39c12"
        else:
            confidence = "LOW"
            confidence_color = "#e74c3c"
        
        self.confidence_card.value_label.setText(confidence)
        self.confidence_card.setStyleSheet(f"""
            QFrame {{
                background-color: {confidence_color};
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }}
        """)
        
        # Update recent average
        self.recent_avg_card.value_label.setText(f"{recent_avg:.1f}")
        
        # Update trend
        diff = ensemble_pred - recent_avg
        if abs(diff) < 2:
            trend = "Neutral"
            trend_color = "#95a5a6"
        elif diff > 2:
            trend = "Up ↗"
            trend_color = "#27ae60"
        else:
            trend = "Down ↘"
            trend_color = "#e74c3c"
        
        self.trend_card.value_label.setText(trend)
        self.trend_card.setStyleSheet(f"""
            QFrame {{
                background-color: {trend_color};
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }}
        """)
