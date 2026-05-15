import sys
from pathlib import Path

# Make `from src.xxx import ...` work when pytest is invoked from any parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
