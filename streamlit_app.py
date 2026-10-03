"""
Streamlit Cloud Entry Point
===========================
Delegates execution to app/streamlit_app.py for instant deployment.
"""

from pathlib import Path
import sys

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the primary Streamlit app
from app.streamlit_app import *
