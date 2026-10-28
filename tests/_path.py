import sys
from pathlib import Path

# The simulator modules are imported as top-level modules (python src/main.py)
SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
