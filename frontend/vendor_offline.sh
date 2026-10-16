#!/usr/bin/env bash
# One-time: pull React + Babel locally so the UI runs with NO internet (air-gapped lab).
# After running, edit index.html to point the three <script> srcs at ./vendor/... .
set -e
cd "$(dirname "$0")"
mkdir -p vendor
curl -L -o vendor/react.production.min.js       https://unpkg.com/react@18/umd/react.production.min.js
curl -L -o vendor/react-dom.production.min.js   https://unpkg.com/react-dom@18/umd/react-dom.production.min.js
curl -L -o vendor/babel.min.js                  https://unpkg.com/@babel/standalone/babel.min.js
echo "Done. Now change index.html script srcs to ./vendor/react.production.min.js etc."
