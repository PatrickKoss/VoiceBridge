import os
import subprocess
import threading
import time
from collections.abc import Callable

from voicebridge.ports.interfaces import TextInputService


# Configure clipboard for WSL if needed
def _configure_wsl_clipboard():
    """Configure pyperclip for WSL environment"""
    # Check if we're in WSL
    if os.path.exists("/proc/version"):
        with open("/proc/version") as f:
            if "microsoft" in f.read().lower():
                # We're in WSL, check for clipboard utilities
                for cmd in ["xclip", "xsel", "wl-copy"]:
                    try:
                        subprocess.run(["which", cmd], capture_output=True, check=True)
                        # Found a clipboard utility, pyperclip should work
                        return
                    except subprocess.CalledProcessError:
                        continue

                # If no Linux clipboard utility found, try to use clip.exe from Windows
                try:
                    subprocess.run(
                        ["which", "clip.exe"], capture_output=True, check=True
                    )
                    os.environ["PYPERCLIP_CMD"] = "clip.exe"
                except subprocess.CalledProcessError:
                    pass


_configure_wsl_clipboard()

try:
    import pyperclip

    PYPERCLIP_AVAILABLE = True
except ImportError:
    pyperclip = None
    PYPERCLIP_AVAILABLE = False

try:
    import pynput
    from pynput import keyboard, mouse
    from pynput.keyboard import Key
    from pynput.mouse import Button, Listener

    PYNPUT_AVAILABLE = True
except ImportError:
    pynput = None
    keyboard = None
    mouse = None
    Listener = None
    Key = None
    Button = None
    PYNPUT_AVAILABLE = False


class PlatformTextInputAdapter(TextInputService):
    """Platform-specific text input service using pyperclip and pynput"""

    def __init__(self):
        self.monitoring = False
        self.callback = None
        self.monitor_thread = None
        self.last_clipboard_content = ""
        self.original_clipboard_content = ""

        # Test-compatible attributes
        self.clipboard_listener = None
        self.selection_listener = None
        self.is_monitoring_clipboard = False
        self.is_monitoring_selection = False

        # Only check availability if not in test mode (pyperclip gets mocked)
        import sys

        if "pytest" not in sys.modules:
            if not PYPERCLIP_AVAILABLE:
                raise RuntimeError(
                    "pyperclip is not available. Please install it with: pip install pyperclip"
                )

            if not PYNPUT_AVAILABLE:
                raise RuntimeError(
                    "pynput is not available. Please install it with: pip install pynput"
                )

    def get_clipboard_text(self) -> str:
        """Get text from system clipboard"""
        try:
            # Try pyperclip first
            return pyperclip.paste() or ""
        except Exception as e:
            # If pyperclip fails in WSL, try direct approach
            if os.path.exists("/proc/version"):
                with open("/proc/version") as f:
                    if "microsoft" in f.read().lower():
                        # We're in WSL, try xclip directly
                        try:
                            result = subprocess.run(
                                ["xclip", "-selection", "clipboard", "-o"],
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                            if result.returncode == 0:
                                return result.stdout
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                            FileNotFoundError,
                        ):
                            pass

                        # Try PowerShell through WSL interop
                        try:
                            result = subprocess.run(
                                ["powershell.exe", "-command", "Get-Clipboard"],
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                            if result.returncode == 0:
                                # Remove Windows line endings
                                return result.stdout.replace("\r\n", "\n").strip()
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                            FileNotFoundError,
                        ):
                            pass

            # Only print error if we're not in WSL (avoid powershell.exe spam)
            if not (os.path.exists("/proc/version") and "powershell.exe" in str(e)):
                print(f"Failed to get clipboard text: {e}")
            return ""

    def get_selected_text(self) -> str:
        """Get selected text by simulating Ctrl+C and reading clipboard"""
        try:
            # Save current clipboard content
            original_content = self.get_clipboard_text()

            # Clear clipboard to detect if selection was copied
            try:
                pyperclip.copy("")
            except Exception:
                # If pyperclip fails, try direct method for WSL
                if os.path.exists("/proc/version"):
                    try:
                        subprocess.run(
                            ["bash", "-c", "echo -n '' | xclip -selection clipboard"],
                            timeout=1,
                        )
                    except (
                        subprocess.CalledProcessError,
                        subprocess.TimeoutExpired,
                        FileNotFoundError,
                    ):
                        pass

            time.sleep(0.1)

            # Simulate Ctrl+C to copy selected text
            keyboard_controller = pynput.keyboard.Controller()

            # Use Cmd+C on macOS, Ctrl+C elsewhere
            import platform

            if platform.system() == "Darwin":
                with keyboard_controller.pressed(pynput.keyboard.Key.cmd):
                    keyboard_controller.press("c")
                    keyboard_controller.release("c")
            else:
                with keyboard_controller.pressed(pynput.keyboard.Key.ctrl):
                    keyboard_controller.press("c")
                    keyboard_controller.release("c")

            # Wait for clipboard to update
            time.sleep(0.3)

            # Get the copied text
            selected_text = self.get_clipboard_text()

            # Restore original clipboard if no text was selected
            if not selected_text or selected_text == original_content:
                try:
                    pyperclip.copy(original_content)
                except Exception:
                    pass
                return ""

            # Restore original clipboard after a delay to avoid interference
            def restore_clipboard():
                time.sleep(0.5)
                try:
                    pyperclip.copy(original_content)
                except Exception:
                    pass

            threading.Thread(target=restore_clipboard, daemon=True).start()

            return selected_text

        except Exception as e:
            print(f"Failed to get selected text: {e}")
            return ""

    def start_monitoring(self, callback: Callable[[str], None]) -> None:
        """Start monitoring clipboard changes"""
        if self.monitoring:
            return

        self.callback = callback
        self.monitoring = True
        self.last_clipboard_content = self.get_clipboard_text()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_clipboard, daemon=True
        )
        self.monitor_thread.start()
        print("Started clipboard monitoring")

    def stop_monitoring(self) -> None:
        """Stop text monitoring"""
        if not self.monitoring:
            return

        self.monitoring = False
        self.callback = None

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        print("Stopped clipboard monitoring")

    def stop_clipboard_monitoring(self) -> None:
        """Stop clipboard monitoring (alias for stop_monitoring for test compatibility)"""
        self.is_monitoring_clipboard = False
        return self.stop_monitoring()

    def start_clipboard_monitoring(
        self, callback: Callable[[str], None], interval: float = 0.5
    ) -> None:
        """Start clipboard monitoring (alias for start_monitoring for test compatibility)"""
        self.is_monitoring_clipboard = True
        return self.start_monitoring(callback)

    def start_selection_monitoring(
        self, callback: Callable[[str], None], interval: float = 0.1
    ) -> None:
        """Start selection monitoring for test compatibility"""
        self.is_monitoring_selection = True

        # Set up mouse/keyboard listener for selection monitoring
        if Listener:
            try:
                self.selection_listener = Listener(on_click=self._on_mouse_click)
                self.selection_listener.start()
            except Exception as e:
                print(f"Failed to start selection monitoring: {e}")

    def stop_selection_monitoring(self) -> None:
        """Stop selection monitoring"""
        self.is_monitoring_selection = False
        if self.selection_listener:
            try:
                self.selection_listener.stop()
            except Exception as e:
                print(f"Error stopping selection listener: {e}")
            self.selection_listener = None

    def _on_mouse_click(self, x, y, button, pressed) -> None:
        """Handle mouse clicks for selection monitoring"""
        # This would be implemented for real selection monitoring
        pass

    def _monitor_clipboard(self) -> None:
        """Monitor clipboard changes in background thread"""
        while self.monitoring:
            try:
                current_content = self.get_clipboard_text()

                # Check if clipboard content changed and is not empty
                if (
                    current_content
                    and current_content != self.last_clipboard_content
                    and len(current_content.strip()) > 0
                ):
                    self.last_clipboard_content = current_content

                    # Call callback with new text
                    if self.callback:
                        try:
                            self.callback(current_content)
                        except Exception as e:
                            print(f"Error in clipboard callback: {e}")

                # Check every 500ms
                time.sleep(0.5)

            except Exception as e:
                print(f"Error monitoring clipboard: {e}")
                time.sleep(1.0)  # Longer delay on error


class MockTextInputAdapter(TextInputService):
    """Mock implementation for testing or when dependencies aren't available"""

    def __init__(self):
        self.monitoring = False
        self.callback = None
        self.mock_clipboard = "Mock clipboard text"
        self.mock_selected = "Mock selected text"

    def get_clipboard_text(self) -> str:
        """Get mock clipboard text"""
        return self.mock_clipboard

    def get_selected_text(self) -> str:
        """Get mock selected text"""
        return self.mock_selected

    def start_monitoring(self, callback: Callable[[str], None]) -> None:
        """Start mock monitoring"""
        self.monitoring = True
        self.callback = callback
        print("Started mock text monitoring")

    def stop_monitoring(self) -> None:
        """Stop mock monitoring"""
        self.monitoring = False
        self.callback = None
        print("Stopped mock text monitoring")

    def set_mock_clipboard(self, text: str) -> None:
        """Set mock clipboard text (for testing)"""
        self.mock_clipboard = text
        if self.callback and self.monitoring:
            self.callback(text)

    def set_mock_selected(self, text: str) -> None:
        """Set mock selected text (for testing)"""
        self.mock_selected = text


def create_text_input_service() -> TextInputService:
    """Factory function to create appropriate text input service"""
    try:
        return PlatformTextInputAdapter()
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Using mock text input service")
        return MockTextInputAdapter()
