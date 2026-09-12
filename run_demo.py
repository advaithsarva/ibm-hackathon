#!/usr/bin/env python3
"""
Quick-start script for The Sentinel Grid.
Run from the repo root:  python run_demo.py
Opens the dashboard in your browser automatically.
"""
import subprocess
import sys
import time
import webbrowser
import os

PORT = 8000
URL = f"http://localhost:{PORT}"

def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   🛡  THE SENTINEL GRID — Disaster Intelligence     ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(f"  Starting API server on  {URL}")
    print("  Press Ctrl+C to stop.\n")

    # Start uvicorn
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", str(PORT), "--reload"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # Give the server a moment then open the browser
    time.sleep(2)
    webbrowser.open(URL)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n  Server stopped.")

if __name__ == "__main__":
    main()
