"""Index the screenshots you already have, keep capturing the screen, and search. Ctrl-C stops capturing.

Settings left out here come from `streamembed config` (defaults: every 5 s, every display, changed screens only).
To keep capturing with no terminal open, use `streamembed start` instead. This script is an alternative to it: while
`streamembed start` captures into the same index, it cannot write there (stop that first, or use another index_dir).
"""
import streamembed as se

SCREENSHOT_FOLDER = "~/Desktop"     # where your screenshots are saved: ~/Desktop on macOS, ~/Pictures/Screenshots on Windows and Ubuntu
INDEX_TYPE = "flat"                 # one of se.available_index_types(); can be changed later without indexing again
QUESTION = "the flight search page with fares to Denver"

se.configure(index_type=INDEX_TYPE)

# Screenshots you already have. Ctrl-C stops the script here; run it again and it continues where it stopped. A folder with no
# screenshots gives zero counts; on macOS, the terminal needs access to ~/Desktop (System Settings > Privacy & Security > Files and
# Folders).
stats = se.build_index_from_directory(SCREENSHOT_FOLDER)
print(stats)
if stats["stopped"]:
    raise SystemExit("Stopped; run it again to continue.")

# From now on, in the background
capture = se.build_index_from_now()

# Search: print the best matches, and open a page with the best 100 to browse by date
for r in se.query_index(QUESTION, top_k=5):
    print(f"{r.score:.3f}  {r.time}  {r.path}")
se.visualize_results(se.query_index(QUESTION, top_k=100))

capture.wait()                      # keep capturing until Ctrl-C
