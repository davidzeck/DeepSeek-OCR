#!/bin/bash

# Simple script to serve the UI with Python HTTP server
# Usage: ./start_ui.sh [port]

PORT=${1:-8080}
UI_DIR="$(dirname "$0")/ui"

if [ ! -d "$UI_DIR" ]; then
    echo "Error: UI directory not found at $UI_DIR"
    exit 1
fi

echo "Starting UI server on port $PORT"
echo "Open http://localhost:$PORT in your browser"
echo ""

cd "$UI_DIR"
python3 -m http.server $PORT

