"""
Application life cycle manager.

Sets up PyQt application environment, initializes logging, Exception handlers,
loads persistent user configurations, and spawns the main window shell.
"""

from __future__ import annotations

import logging
import sys
from PyQt6.QtWidgets import QApplication

from settings.settings_manager import SettingsManager
from ui.main_window import MainWindow


class AirWritingApplication:
    """
    Coordinates application startup, configuration loading, and teardown.
    """

    def __init__(self):
        self._init_logging()
        self.settings = SettingsManager()
        self.app = None
        self.main_window = None

    def _init_logging(self) -> None:
        """Configures console logging for diagnostic reporting."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("AirWritingApp")
        self.logger.info("Initializing system loggers...")

    def run(self) -> int:
        """
        Executes startup lifecycle and runs the main event loop.

        Returns:
            Exit code integer.
        """
        self.logger.info("Starting Air Writing Desktop Application...")
        
        # Create Qt Application instance
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Air Writing")
        self.app.setOrganizationName("Antigravity CV")
        
        # Wire global uncaught exception handlers to avoid silent crashes
        sys.excepthook = self._handle_exception

        # Create and display the main interface
        self.main_window = MainWindow(self.settings)
        self.main_window.show()
        
        # Execute main event loop
        exit_code = self.app.exec()
        self.logger.info("System loop completed with exit code: %d", exit_code)
        return exit_code

    def _handle_exception(self, exc_type, exc_value, exc_traceback) -> None:
        """Custom global uncaught exception handler to prevent silent aborts."""
        self.logger.error(
            "Uncaught system exception encountered!",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        # We let the default handler run if necessary, but logging is critical
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
