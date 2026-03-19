import sys
from pathlib import Path

# Make the repo root importable so smoke_test.py can `from train_gpt import ...`
sys.path.insert(0, str(Path(__file__).parent))
