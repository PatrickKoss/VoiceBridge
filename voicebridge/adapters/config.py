import json
from pathlib import Path

from voicebridge.domain.models import WhisperConfig
from voicebridge.ports.interfaces import ConfigRepository, ProfileRepository


class FileConfigRepository(ConfigRepository):
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = config_dir / "config.json"
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            # If we can't create the directory, we'll work with defaults
            pass

    def load(self) -> WhisperConfig:
        defaults = WhisperConfig()
        if not self.config_file.exists():
            return defaults

        try:
            with open(self.config_file) as f:
                data = json.load(f)
                return WhisperConfig.from_dict({**defaults.to_dict(), **data})
        except Exception:
            return defaults

    def save(self, config: WhisperConfig) -> None:
        self._ensure_config_dir()
        try:
            with open(self.config_file, "w") as f:
                json.dump(config.to_dict(), f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Could not save config: {e}") from e


class FileProfileRepository(ProfileRepository):
    def __init__(self, profiles_dir: Path):
        self.profiles_dir = profiles_dir
        self._ensure_profiles_dir()

    def _ensure_profiles_dir(self):
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, name: str) -> Path:
        return self.profiles_dir / f"{name}.json"

    def save_profile(self, name: str, config: WhisperConfig) -> None:
        self._ensure_profiles_dir()
        profile_path = self._profile_path(name)
        try:
            with open(profile_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Could not save profile '{name}': {e}") from e

    def load_profile(self, name: str) -> WhisperConfig:
        profile_path = self._profile_path(name)
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile '{name}' not found")

        try:
            with open(profile_path) as f:
                data = json.load(f)
                return WhisperConfig.from_dict(data)
        except Exception as e:
            raise RuntimeError(f"Could not load profile '{name}': {e}") from e

    def list_profiles(self) -> list[str]:
        if not self.profiles_dir.exists():
            return []
        return [f.stem for f in self.profiles_dir.glob("*.json")]

    def delete_profile(self, name: str) -> bool:
        profile_path = self._profile_path(name)
        if profile_path.exists():
            profile_path.unlink()
            return True
        return False
