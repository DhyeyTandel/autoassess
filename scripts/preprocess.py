#!/usr/bin/env python3
"""scripts/preprocess.py — standalone preprocessing script.

Thin wrapper around `autoassess-prep` for use before installing the package.
"""

import sys
from pathlib import Path

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoassess.data.prepare import main  # noqa: E402

if __name__ == "__main__":
    main()
