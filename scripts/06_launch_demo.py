#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 6: Launch Demo
======================================
Starts both the Gradio UI and (optionally) the FastAPI server.

Usage:
    python scripts/06_launch_demo.py              # Gradio only
    python scripts/06_launch_demo.py --api         # Gradio + FastAPI
    python scripts/06_launch_demo.py --api-only    # FastAPI only
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import API_PORT, GRADIO_PORT


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 6: Launch SafeSteer-IN demo")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Also start the FastAPI server in a background process",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the FastAPI server (no Gradio)",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio sharing link"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeSteer-IN  ·  Step 6: Launch Demo")
    print("=" * 60)

    if args.api_only:
        print(f"\n  Starting FastAPI server on port {API_PORT} …")
        import uvicorn

        sys.path.insert(0, str(ROOT))
        uvicorn.run("api:app", host="0.0.0.0", port=API_PORT, reload=False)
        return

    if args.api:
        print(f"\n  Starting FastAPI server in background on port {API_PORT} …")
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(API_PORT),
            ],
            cwd=str(ROOT),
        )

    print(f"\n  Starting Gradio UI on port {GRADIO_PORT} …")
    print(f"  Open: http://localhost:{GRADIO_PORT}\n")

    from app import create_demo

    demo = create_demo()
    demo.launch(
        server_port=GRADIO_PORT,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
