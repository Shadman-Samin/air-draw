"""
Persistent Settings Manager.

Handles loading and saving configuration values from/to a local JSON file.
Ensures default fallback values exist for any missing or corrupted keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from settings.defaults import DEFAULT_SETTINGS


class SettingsManager:
    """
    Manages loading, updating, and saving application configurations.

    Automatically handles deep merging of loaded configurations with defaults
    to prevent exceptions due to missing settings.
    """

    def __init__(self, filename: str = "settings.json"):
        # Store in user's profile folder under .airdraw
        self._app_dir = Path.home() / ".airdraw"
        self._settings_path = self._app_dir / filename
        self._settings: dict[str, Any] = {}
        
        self.load()

    @property
    def all_settings(self) -> dict[str, Any]:
        """Get copy of all current settings."""
        return self._settings.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a nested setting using a period-separated key (e.g. 'camera.width').
        """
        parts = key.split(".")
        val = self._settings
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        """
        Set a nested setting using a period-separated key (e.g. 'camera.width').
        Saves automatically upon setting modification.
        """
        parts = key.split(".")
        val = self._settings
        
        # Traverse down to the second to last part
        for part in parts[:-1]:
            if part not in val or not isinstance(val[part], dict):
                val[part] = {}
            val = val[part]
            
        # Set the value of the last part
        last_part = parts[-1]
        val[last_part] = value
        
        self.save()

    def load(self) -> None:
        """Load settings from file, merging with default fallback values."""
        # Start with defaults
        self._settings = self._deep_copy(DEFAULT_SETTINGS)
        
        if not self._settings_path.exists():
            # No settings file exists yet, save current default set
            self.save()
            return
            
        try:
            with open(self._settings_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self._deep_merge(self._settings, loaded)
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Silently fallback to default settings if file is corrupt
            print(f"[Warning] Failed to load settings from {self._settings_path}: {e}")
            self.save()  # Repair by rewriting defaults

    def save(self) -> None:
        """Save settings persistently to disk."""
        try:
            self._app_dir.mkdir(parents=True, exist_ok=True)
            with open(self._settings_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except (IOError, OSError) as e:
            print(f"[Error] Failed to save settings to {self._settings_path}: {e}")

    def reset_to_defaults(self) -> None:
        """Reset current configurations to system defaults."""
        self._settings = self._deep_copy(DEFAULT_SETTINGS)
        self.save()

    def _deep_copy(self, data: Any) -> Any:
        """Simple deep copy implementation to avoid using copy module."""
        if isinstance(data, dict):
            return {k: self._deep_copy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deep_copy(v) for v in data]
        return data

    def _deep_merge(self, base: dict[str, Any], overlay: dict[str, Any]) -> None:
        """Recursively merges overlay dictionary into base dictionary."""
        for k, v in overlay.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = self._deep_copy(v)
