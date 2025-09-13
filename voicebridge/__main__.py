#!/usr/bin/env python3

import sys
from pathlib import Path

# Add parent directory to Python path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.main import main

if __name__ == "__main__":
    main()
