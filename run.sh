#!/bin/bash
# Parakeet Web UI 起動スクリプト

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "🦜 Parakeet 日本語音声書き起こし Web UI"
echo "========================================"

# 仮想環境を有効化
if [ -d "venv" ]; then
    echo "📦 仮想環境を有効化中..."
    source venv/bin/activate
else
    echo "❌ 仮想環境が見つかりません。"
    echo "   python3 -m venv venv で作成してください。"
    exit 1
fi

# モデルファイルの確認
if [ ! -f "parakeet-tdt_ctc-0.6b-ja.nemo" ]; then
    echo "❌ モデルファイルが見つかりません:"
    echo "   parakeet-tdt_ctc-0.6b-ja.nemo"
    exit 1
fi

echo ""
echo "🚀 アプリを起動中..."
echo "   ブラウザで http://localhost:7860 を開いてください"
echo ""

python app.py
