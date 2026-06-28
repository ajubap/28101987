"""Convenience launcher for the dashboard.

    python run_dashboard.py            # http://127.0.0.1:8050
    python run_dashboard.py --port 8060 --host 0.0.0.0
"""

from __future__ import annotations

import argparse

from dashboard.app import app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
