"""Makes the repository's paths.py (data and output locations) importable from figures/source."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import *  # noqa: E402,F401,F403
