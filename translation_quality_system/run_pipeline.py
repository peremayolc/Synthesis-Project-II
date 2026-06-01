import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_unitron_verification import main  # noqa: E402

if __name__ == "__main__":
    main()
