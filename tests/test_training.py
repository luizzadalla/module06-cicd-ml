import subprocess
import sys
from pathlib import Path

def test_train_runs_and_saves_model():
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path("model.pkl").exists()