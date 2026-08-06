#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Launch the Bokeh corner picker app.")
    parser.add_argument("stiched_image", help="Path to the stitched image.")
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for the Bokeh server (default: Bokeh's default, usually 5006).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.stiched_image):
        print(f"Error: Directory '{args.stiched_image}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            "bokeh", "serve", os.path.join(scripts_dir, "corner_picker.py"),
            "--show",
        ]
        if args.port is not None:
            cmd.extend(["--port", str(args.port)])
        cmd.extend(["--args", args.stiched_image])
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching Bokeh app: {e}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
