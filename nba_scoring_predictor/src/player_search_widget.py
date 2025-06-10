"""
Enhanced Player Search Widget with Autocomplete
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, 
    QPushButton, QCompleter, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QStringListModel
from PyQt5.QtGui import QFont
from typing import List, Optional
import difflib

class PlayerSearchWidget(QWidget):
    """Enhanced player search widget with autocomplete functionality."""
    
    player_selected = pyqtSignal(str)  # Signal emitted when player is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_players = []
        self.filtered_players = []
        self.completer = None
        self.completer_model = QStringListModel()
        
        self.init_ui()
        self.setup_completer()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("🔍 Search Player")
        title_label.setFont(QFont("Arial", 11, QFont.Bold))
        title_label.setStyleSheet("color: white; margin-bottom: 5px;")
        layout.addWidget(title_label)
        
        # Search input frame
        search_frame = QFrame()
        search_frame.setFrameStyle(QFrame.StyledPanel)
        search_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        search_layout = QHBoxLayout(search_frame)
        search_layout.setContentsMargins(5, 5, 5, 5)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type player name... (e.g., LeBron James)")
        self.search_input.setMinimumHeight(35)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495e;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 8px 12px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: #2c3e50;
            }
        """)
        
        # Connect text change signal
        self.search_input.textChanged.connect(self.on_text_changed)
        self.search_input.returnPressed.connect(self.on_enter_pressed)
        
        # Search button
        self.search_button = QPushButton("🔍")
        self.search_button.setMinimumSize(35, 35)
        self.search_button.setMaximumSize(35, 35)
        self.search_button.clicked.connect(self.on_search_clicked)
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        
        layout.addWidget(search_frame)
        
        # Info label
        self.info_label = QLabel("Start typing to see player suggestions...")
        self.info_label.setStyleSheet("color: #95a5a6; font-size: 10px; margin-top: 5px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
    
    def setup_completer(self):
        """Setup the autocomplete functionality."""
        self.completer = QCompleter()
        self.completer.setModel(self.completer_model)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)  # Allow matching anywhere in the string
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.setMaxVisibleItems(10)
        
        # Style the popup
        popup = self.completer.popup()
        popup.setStyleSheet("""
            QListView {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #3498db;
                selection-background-color: #3498db;
                selection-color: white;
                font-size: 11px;
                padding: 2px;
            }
            QListView::item {
                padding: 5px;
                border-bottom: 1px solid #34495e;
            }
            QListView::item:hover {
                background-color: #34495e;
            }
        """)
        
        # Connect signals
        self.completer.activated.connect(self.on_completion_selected)
        self.search_input.setCompleter(self.completer)
    
    def update_player_list(self, players: List[str]):
        """Update the list of available players."""
        self.all_players = sorted(list(set(players)))  # Remove duplicates and sort
        self.update_completer_model()
        
        if self.all_players:
            self.info_label.setText(f"✅ {len(self.all_players)} players available for search")
            self.search_input.setEnabled(True)
            self.search_button.setEnabled(True)
        else:
            self.info_label.setText("❌ No players available. Load a model first.")
            self.search_input.setEnabled(False)
            self.search_button.setEnabled(False)
    
    def update_completer_model(self, filter_text: str = ""):
        """Update the completer model with filtered players."""
        if not filter_text:
            filtered_players = self.all_players
        else:
            # Use fuzzy matching for better search results
            filtered_players = self.fuzzy_search(filter_text, self.all_players)
        
        self.completer_model.setStringList(filtered_players)
    
    def fuzzy_search(self, query: str, players: List[str], max_results: int = 20) -> List[str]:
        """Perform fuzzy search on player names."""
        if not query:
            return players[:max_results]
        
        query_lower = query.lower()
        
        # Exact matches first
        exact_matches = [p for p in players if query_lower in p.lower()]
        
        # If we have enough exact matches, return them
        if len(exact_matches) >= max_results:
            return exact_matches[:max_results]
        
        # Add fuzzy matches
        remaining_players = [p for p in players if p not in exact_matches]
        fuzzy_matches = difflib.get_close_matches(
            query, remaining_players, n=max_results - len(exact_matches), cutoff=0.3
        )
        
        return exact_matches + fuzzy_matches
    
    def on_text_changed(self, text: str):
        """Handle text change in search input."""
        if len(text) >= 2:  # Start suggesting after 2 characters
            self.update_completer_model(text)
            
            # Show number of matches
            matches = self.fuzzy_search(text, self.all_players)
            if matches:
                self.info_label.setText(f"🔍 {len(matches)} matches found")
            else:
                self.info_label.setText("❌ No matches found")
        elif text:
            self.info_label.setText("Type at least 2 characters...")
        else:
            self.info_label.setText("Start typing to see player suggestions...")
            self.update_completer_model()
    
    def on_completion_selected(self, text: str):
        """Handle completion selection."""
        self.search_input.setText(text)
        self.select_player(text)
    
    def on_enter_pressed(self):
        """Handle Enter key press."""
        text = self.search_input.text().strip()
        if text:
            # Try to find exact match or closest match
            matches = self.fuzzy_search(text, self.all_players, 1)
            if matches:
                selected_player = matches[0]
                self.search_input.setText(selected_player)
                self.select_player(selected_player)
            else:
                self.info_label.setText("❌ Player not found. Try a different name.")
    
    def on_search_clicked(self):
        """Handle search button click."""
        self.on_enter_pressed()
    
    def select_player(self, player_name: str):
        """Select a player and emit signal."""
        if player_name in self.all_players:
            self.info_label.setText(f"✅ Selected: {player_name}")
            self.player_selected.emit(player_name)
        else:
            self.info_label.setText(f"❌ Player '{player_name}' not found in database")
    
    def clear_search(self):
        """Clear the search input."""
        self.search_input.clear()
        self.info_label.setText("Start typing to see player suggestions...")
    
    def set_selected_player(self, player_name: str):
        """Set the selected player programmatically."""
        if player_name in self.all_players:
            self.search_input.setText(player_name)
            self.info_label.setText(f"✅ Selected: {player_name}")
