"""
Simple webhook service that matches the CLI expectations.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class SimpleWebhookService:
    """Simple webhook service for CLI use."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = config_dir / "webhooks.json"
        self._webhooks = self._load_webhooks()

    def _load_webhooks(self) -> list[dict[str, Any]]:
        """Load webhook configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_webhooks(self):
        """Save webhook configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self._webhooks, f, indent=2)

    def add_webhook(self, webhook_config: dict[str, Any]) -> bool:
        """Add a webhook configuration."""
        try:
            # Check if webhook already exists
            url = webhook_config["url"]
            for webhook in self._webhooks:
                if webhook["url"] == url:
                    # Update existing webhook
                    webhook.update(webhook_config)
                    webhook["added_at"] = datetime.now().isoformat()
                    self._save_webhooks()
                    return True

            # Add new webhook
            webhook_config["added_at"] = datetime.now().isoformat()
            webhook_config["status"] = "active"
            webhook_config["last_used"] = "Never"
            self._webhooks.append(webhook_config)
            self._save_webhooks()
            return True
        except Exception:
            return False

    def remove_webhook(self, url: str) -> bool:
        """Remove a webhook by URL."""
        try:
            original_length = len(self._webhooks)
            self._webhooks = [w for w in self._webhooks if w["url"] != url]

            if len(self._webhooks) < original_length:
                self._save_webhooks()
                return True
            return False
        except Exception:
            return False

    def list_webhooks(self) -> list[dict[str, Any]]:
        """List all configured webhooks."""
        return self._webhooks.copy()

    def test_webhook(self, url: str, test_payload: dict[str, Any]) -> dict[str, Any]:
        """Test a webhook with sample data."""
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "error": "requests library not available - install with: pip install requests",
                "status_code": 0,
                "response_time": 0,
            }

        try:
            start_time = time.time()

            headers = {
                "Content-Type": "application/json",
                "User-Agent": "VoiceBridge-Webhook-Test/1.0",
            }

            # Find webhook config for timeout and secret
            webhook_config = next((w for w in self._webhooks if w["url"] == url), {})
            timeout = webhook_config.get("timeout", 30)
            secret = webhook_config.get("secret")

            if secret:
                headers["X-Webhook-Secret"] = secret

            response = requests.post(
                url, json=test_payload, headers=headers, timeout=timeout
            )

            response_time = time.time() - start_time

            # Update webhook status
            for webhook in self._webhooks:
                if webhook["url"] == url:
                    webhook["last_used"] = datetime.now().isoformat()
                    webhook["status"] = (
                        "active" if response.status_code < 400 else "error"
                    )
            self._save_webhooks()

            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response_time": response_time,
                "response_data": response.text[:200]
                if len(response.text) <= 200
                else response.text[:200] + "...",
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "status_code": 0,
                "response_time": timeout,
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection failed",
                "status_code": 0,
                "response_time": 0,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": 0,
                "response_time": 0,
            }
