import sys
from pathlib import Path

# Make `from src.xxx import ...` work when pytest is invoked from a parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
