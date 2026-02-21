import subprocess
import sys
import re

THRESHOLD = 0.99

out = subprocess.check_output([sys.executable, "evaluate.py"], text=True)
acc = float(re.search(r"accuracy=(\d+\.\d+)", out).group(1))

if acc < THRESHOLD:
    print(f"FAIL accuracy {acc:.4f} < {THRESHOLD}")
    sys.exit(1)

print(f"PASS accuracy {acc:.4f} >= {THRESHOLD}")