"""Search your own screen history on your laptop: index your screenshots, capture the screen every 5 s, and ask.

    python run.py                    # your screenshot folder: ~/Desktop on macOS, ~/Pictures/Screenshots elsewhere
    python run.py --folder DIR
"""
import argparse
import platform
from pathlib import Path

import streamembed as se

DEFAULT = "~/Desktop" if platform.system() == "Darwin" else "~/Pictures/Screenshots"

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--folder", default=DEFAULT, help=f"the screenshots you already have (default {DEFAULT})")
a = ap.parse_args()

capture = None
try:
    if Path(a.folder).expanduser().is_dir():
        se.build_index_from_directory(a.folder)
    capture = se.build_index_from_now()
    print("Capturing the screen every 5 s.")
except RuntimeError as e:            # the streamembed service is already indexing: search what it has
    print(e)

try:
    while question := input("\nDescribe a past screen (Enter to quit): ").strip():
        results = se.query_index(question)
        for r in results[:5]:
            print(f"{r.time}  {r.path}")
        if results:
            se.visualize_results(results)
except (KeyboardInterrupt, EOFError):
    pass
finally:
    if capture:
        capture.stop()
