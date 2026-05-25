"""
Air Writing (Virtual Pen) Application — Entry Point.

Instantiates the global application controller and runs the PyQt window lifecycle.
"""

from __future__ import annotations

import sys
from app.application import AirWritingApplication


def main() -> None:
    """Main program entry point."""
    try:
        app = AirWritingApplication()
        sys.exit(app.run())
    except Exception as e:
        print(f"[Fatal] Unhandled startup exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

