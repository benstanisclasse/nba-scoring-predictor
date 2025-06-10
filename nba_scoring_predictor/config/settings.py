# -*- coding: utf-8 -*-
"""
Configuration settings for NBA Scoring Predictor
"""

# Database settings
DATABASE_PATH = "data/nba_data.db"

# Model settings
DEFAULT_SEASONS = ['2022-23', '2023-24', '2024-25']
MIN_GAMES_PLAYED = 10
OPTIMIZATION_TRIALS = 50

# Feature engineering settings
ROLLING_WINDOWS = [3, 5, 10, 15, 20]
EWM_ALPHAS = [0.1, 0.3, 0.5]
MAX_DAYS_REST = 10

# Model configurations
MODEL_CONFIGS = {
    'xgboost': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [20, 31, 50, 100, 200],
        'subsample': [0.6, 0.8, 0.9, 1.0]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    }
}

# GUI settings
WINDOW_SIZE = "1200x800"
THEME = "dark"
COLORS = {
    'primary': '#1f538d',
    'secondary': '#14375e',
    'accent': '#f39c12',
    'background': '#2c3e50',
    'text': '#ecf0f1'
}

# Validation settings
VALIDATION_CONFIG = {
    'min_sample_size': 100,
    'significance_level': 0.05,
    'confidence_intervals': [0.68, 0.95],  # 1 and 2 standard deviations
    'betting_vig': 0.045,  # 4.5% sportsbook commission
    'kelly_fraction': 0.25,  # Fractional Kelly betting
    'backtest_seasons': ['2022-23', '2023-24'],
    'validation_split': 0.8,  # 80% train, 20% test
}

# Report settings
REPORT_CONFIG = {
    'output_dir': 'reports/validation_reports',
    'include_plots': True,
    'plot_format': 'png',
    'generate_html': True,
    'generate_pdf': False
}
