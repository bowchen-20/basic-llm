import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# train.py has no `if __name__ == "__main__"` guard — it runs training on import.
# It is never imported by this suite; if it ever needs testing, do it via
# subprocess.run([sys.executable, "train.py"], ...) in an isolated tmp_path.
