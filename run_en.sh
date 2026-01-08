#!/bin/bash
# English Speech-to-Text Web UI Launch Script
# * Uses CTC decoder for WSL2 compatibility (avoids CUDA Graphs)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "🦜 Nemotron English Speech-to-Text Web UI"
echo "========================================"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found."
    echo "   Create with: python3 -m venv venv"
    exit 1
fi

# Check model file
if [ ! -f "nemotron-speech-streaming-en-0.6b.nemo" ]; then
    echo "❌ Model file not found:"
    echo "   nemotron-speech-streaming-en-0.6b.nemo"
    exit 1
fi

echo ""
echo "🚀 Starting app..."
echo "   Open http://localhost:7861 in your browser"
echo ""

python app_simple_en.py
