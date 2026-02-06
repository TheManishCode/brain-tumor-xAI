#!/usr/bin/env python3
"""
MidLens — Brain Tumor Classification
=====================================
Single entry point.  Run with:

    python run.py            # production (waitress)
    python run.py --debug    # development (Flask debug server)
"""

import argparse

from server.app import create_app
from server.config import SETTINGS


def main() -> None:
    parser = argparse.ArgumentParser(description="MidLens API Server")
    parser.add_argument("--host", default=SETTINGS.host, help="Bind address")
    parser.add_argument("--port", type=int, default=SETTINGS.port, help="Port")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args()

    app = create_app()

    # --- Banner ---
    print(f"\n{'=' * 60}")
    print(f"  {SETTINGS.service_name} — Brain Tumor Classification")
    print(f"{'=' * 60}")
    print(f"  Version : {SETTINGS.version}")
    print(f"  URL     : http://localhost:{args.port}")
    print(f"{'=' * 60}\n")

    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        try:
            from waitress import serve
            serve(app, host=args.host, port=args.port, threads=4)
        except ImportError:
            app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
