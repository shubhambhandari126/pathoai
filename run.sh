#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"
export PYTHONPATH=.:backend
python3 backend/server.py --host 0.0.0.0 --port 8000 --cases 24 --slides 3 --demo-train-steps 0
