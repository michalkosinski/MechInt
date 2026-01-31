#!/usr/bin/env python3
"""
Server Watchdog - Auto-restarts web dashboard if it crashes.

Port 8016 is used for this project (MechInt).

Usage:
    python server_watchdog.py &
"""

import time
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

# Configuration
WORKSPACE = Path(__file__).parent.absolute()
PROJECT_NAME = "MechInt"
SERVER_PORT = 8016
SERVER_HOST = "127.0.0.1"

# Server command
WEB_SCRIPT = "web_dashboard.py"

# Log files
WATCHDOG_LOG = f"/tmp/{PROJECT_NAME}_watchdog.log"
SERVER_LOG = f"/tmp/{PROJECT_NAME}_server.log"

# Health check URL
URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Timing
CHECK_INTERVAL_SEC = 10
STARTUP_GRACE_SEC = 3


def log(msg: str) -> None:
    """Write timestamped message to watchdog log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")


def is_server_up() -> bool:
    """Check if the server is responding."""
    try:
        with urlopen(URL, timeout=3) as resp:
            return 200 <= resp.status < 300
    except (URLError, HTTPError):
        return False
    except Exception as e:
        log(f"Health check error: {e}")
        return False


def kill_existing_server() -> None:
    """Kill any existing server process on our port."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{WEB_SCRIPT}.*{SERVER_PORT}"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(["kill", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log(f"Killed existing server process {pid}")
                except:
                    pass
    except Exception as e:
        log(f"Kill attempt failed (ignored): {e}")


def start_server() -> None:
    """Start the web server."""
    kill_existing_server()

    python_exec = sys.executable
    server_out = open(SERVER_LOG, "a", buffering=1)

    try:
        subprocess.Popen(
            [python_exec, "-u", WEB_SCRIPT, "--port", str(SERVER_PORT)],
            cwd=str(WORKSPACE),
            stdout=server_out,
            stderr=server_out
        )
        log(f"Started server: {python_exec} {WEB_SCRIPT} --port {SERVER_PORT}")
        log(f"Server URL: {URL}")
    except Exception as e:
        log(f"Failed to start server: {e}")


def main():
    """Main watchdog loop."""
    log(f"{'=' * 60}")
    log(f"Watchdog started for {PROJECT_NAME}")
    log(f"Server port: {SERVER_PORT}")
    log(f"Server URL: {URL}")
    log(f"Server log: {SERVER_LOG}")
    log(f"{'=' * 60}")

    # Ensure initial startup
    if not is_server_up():
        log("Server down at start; launching...")
        start_server()
        time.sleep(STARTUP_GRACE_SEC)

    # Main loop
    while True:
        try:
            if not is_server_up():
                log("Server not responding; restarting...")
                start_server()
                time.sleep(STARTUP_GRACE_SEC)
            time.sleep(CHECK_INTERVAL_SEC)
        except KeyboardInterrupt:
            log("Watchdog stopping (KeyboardInterrupt)")
            break
        except Exception as e:
            log(f"Watchdog loop error: {e}")
            time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
