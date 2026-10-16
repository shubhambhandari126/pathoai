#!/usr/bin/env python3
"""PathQA command-line entry point — one command to run everything.

    pathqa serve            # start the local app + open the browser (default)
    pathqa demo   [...]     # patch-level QA demo (see demo.py --help)
    pathqa slide-demo [...] # whole-slide QA demo (see slide_demo.py --help)
    pathqa train  [...]     # train on PatchCamelyon (see train_pcam.py --help)

`serve` fine-tunes a small demo model on first boot so the UI works out of the
box; point it at a trained checkpoint with  PATHQA_MODEL=model.pt pathqa serve.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _run_script_main(module_name: str, argv):
    """Call a top-level script's main() with a synthesized argv."""
    sys.argv = [module_name] + list(argv)
    mod = __import__(module_name)
    mod.main()


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "serve"
    rest = args[1:]

    if cmd in ("serve", "app", "start"):
        import argparse
        ap = argparse.ArgumentParser(prog="pathqa serve")
        ap.add_argument("--port", type=int, default=8000)
        ap.add_argument("--host", default="127.0.0.1")
        ap.add_argument("--no-open", action="store_true", help="don't open the browser")
        ap.add_argument("--cases", type=int, default=200)
        ap.add_argument("--slides", type=int, default=6)
        ap.add_argument("--slide-grid", type=int, default=14)
        a = ap.parse_args(rest)
        from server import serve
        serve(host=a.host, port=a.port, open_browser=not a.no_open,
              n_cases=a.cases, n_slides=a.slides, slide_grid=a.slide_grid)
    elif cmd == "demo":
        _run_script_main("demo", rest)
    elif cmd in ("slide-demo", "slides"):
        _run_script_main("slide_demo", rest)
    elif cmd == "train":
        _run_script_main("train_pcam", rest)
    elif cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
