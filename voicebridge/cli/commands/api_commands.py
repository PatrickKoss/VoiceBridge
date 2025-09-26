import threading
import time

import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
)


class APICommands(BaseCommands):
    """Commands for managing the VoiceBridge API server."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._server_thread = None
        self._server_process = None

    def api_status(self):
        """Show API server status."""
        try:
            import requests

            # Check if server is running on default port
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    typer.echo("API Server Status: Running")
                    typer.echo("  URL: http://localhost:8000")
                    typer.echo(f"  Status: {health_data.get('status', 'unknown')}")
                    typer.echo(
                        f"  Timestamp: {health_data.get('timestamp', 'unknown')}"
                    )

                    # Service availability
                    services = health_data.get("services", {})
                    typer.echo("\nService Availability:")
                    for service, available in services.items():
                        status = "✓ Available" if available else "✗ Unavailable"
                        typer.echo(f"  {service}: {status}")
                else:
                    typer.echo("API Server Status: Error")
                    typer.echo(f"  HTTP Status: {response.status_code}")

            except requests.exceptions.ConnectionError:
                typer.echo("API Server Status: Not Running")
                typer.echo("  Use 'voicebridge api start' to start the server")

            except requests.exceptions.Timeout:
                typer.echo("API Server Status: Unresponsive")
                typer.echo("  Server may be starting or overloaded")

        except ImportError:
            display_error(
                "requests library not available. Install with: pip install requests"
            )
        except Exception as e:
            display_error(f"Error checking API status: {e}")

    def api_start(
        self,
        host: str = "localhost",
        port: int = 8000,
        workers: int = 1,
        background: bool = False,
    ):
        """Start the API server."""
        try:
            # Check if FastAPI/uvicorn are available
            try:
                import uvicorn

                from voicebridge.adapters.api_endpoints import create_api_app
            except ImportError:
                display_error("FastAPI and uvicorn are required for the API server.")
                typer.echo("Install with: pip install 'fastapi' 'uvicorn[standard]'")
                return

            # Check if server is already running
            try:
                import requests

                response = requests.get(f"http://{host}:{port}/health", timeout=2)
                if response.status_code == 200:
                    display_error(f"API server is already running on {host}:{port}")
                    return
            except (ImportError, requests.exceptions.RequestException):
                # Server not running or requests not available, continue
                pass

            display_info(f"Starting API server on {host}:{port}...")

            # Create API app with available services
            services = self._get_api_services()
            app = create_api_app(**services)

            if background:
                # Start server in background thread
                def run_server():
                    uvicorn.run(
                        app, host=host, port=port, workers=workers, log_level="info"
                    )

                self._server_thread = threading.Thread(target=run_server, daemon=True)
                self._server_thread.start()

                # Wait a moment to see if it starts successfully
                time.sleep(2)

                try:
                    import requests

                    response = requests.get(f"http://{host}:{port}/health", timeout=5)
                    if response.status_code == 200:
                        display_progress(
                            f"API server started in background on {host}:{port}",
                            finished=True,
                        )
                        typer.echo(f"  Health check: http://{host}:{port}/health")
                        typer.echo(f"  API docs: http://{host}:{port}/docs")
                    else:
                        display_error("Server started but health check failed")
                except Exception:
                    display_error("Server may have started but health check failed")
            else:
                # Start server in foreground
                typer.echo(f"API server will start on http://{host}:{port}")
                typer.echo(f"API documentation available at: http://{host}:{port}/docs")
                typer.echo("Press Ctrl+C to stop the server")

                try:
                    uvicorn.run(
                        app, host=host, port=port, workers=workers, log_level="info"
                    )
                except KeyboardInterrupt:
                    display_info("API server stopped")

        except Exception as e:
            display_error(f"Failed to start API server: {e}")

    def api_stop(self, port: int = 8000):
        """Stop the API server."""
        try:
            # For background threads
            if self._server_thread and self._server_thread.is_alive():
                display_info("Stopping background API server...")
                # Note: This is a limitation - we can't cleanly stop uvicorn from a thread
                display_info("Background server will stop when the main process exits")
                return

            # For external processes, try to find and stop them
            try:
                import psutil

                # Find processes using the port
                connections = psutil.net_connections()
                pids = []

                for conn in connections:
                    if conn.laddr.port == port and conn.status == "LISTEN":
                        pids.append(conn.pid)

                if pids:
                    for pid in pids:
                        try:
                            process = psutil.Process(pid)
                            if "uvicorn" in " ".join(process.cmdline()).lower():
                                display_info(
                                    f"Stopping API server process (PID: {pid})"
                                )
                                process.terminate()

                                # Wait for graceful shutdown
                                try:
                                    process.wait(timeout=5)
                                    display_progress(
                                        "API server stopped", finished=True
                                    )
                                except psutil.TimeoutExpired:
                                    process.kill()
                                    display_progress(
                                        "API server forcibly stopped", finished=True
                                    )
                        except psutil.NoSuchProcess:
                            continue
                else:
                    display_info(f"No API server found running on port {port}")

            except ImportError:
                display_error("psutil library not available for process management")
                display_info("Install with: pip install psutil")
                display_info(
                    "Or manually stop the server with Ctrl+C if running in foreground"
                )

        except Exception as e:
            display_error(f"Error stopping API server: {e}")

    def api_info(self):
        """Show API server information and endpoints."""
        typer.echo("VoiceBridge API Server Information")
        typer.echo("=" * 40)

        typer.echo("\nMain Endpoints:")
        typer.echo("  POST /transcribe           - Submit audio for transcription")
        typer.echo("  GET  /transcribe/{id}      - Get transcription result")
        typer.echo("  GET  /progress/{id}        - Get operation progress")
        typer.echo("  GET  /operations           - List all operations")
        typer.echo("  DELETE /operations/{id}    - Cancel an operation")
        typer.echo("  POST /webhook/register     - Register webhook URL")
        typer.echo("  GET  /health               - Health check")

        typer.echo("\nUsage Examples:")
        typer.echo("  # Start server")
        typer.echo("  voicebridge api start")
        typer.echo("  ")
        typer.echo("  # Start server in background")
        typer.echo("  voicebridge api start --background")
        typer.echo("  ")
        typer.echo("  # Start on custom port")
        typer.echo("  voicebridge api start --port 8080")
        typer.echo("  ")
        typer.echo("  # Check server status")
        typer.echo("  voicebridge api status")

        typer.echo("\nAPI Documentation:")
        typer.echo("  When server is running, visit:")
        typer.echo("  http://localhost:8000/docs      - Interactive API docs")
        typer.echo("  http://localhost:8000/redoc     - ReDoc documentation")

    def _get_api_services(self) -> dict:
        """Get available services for the API."""
        return {
            "transcription_service": self.transcription_orchestrator,
            "vocabulary_service": self.vocabulary_service,
            "post_processing_service": self.postprocessing_service,
            "webhook_service": self.webhook_service,
            "progress_service": self.progress_service,
        }
