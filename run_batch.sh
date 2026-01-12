#!/bin/bash
# Batch Speech-to-Text 起動スクリプト
# 複数ファイル一括書き起こし（日本語/English）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "🦜 Batch Speech-to-Text (日本語/English)"
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
JA_MODEL="parakeet-tdt_ctc-0.6b-ja.nemo"
EN_MODEL="nemotron-speech-streaming-en-0.6b.nemo"

if [ ! -f "$JA_MODEL" ] && [ ! -f "$EN_MODEL" ]; then
    echo "❌ モデルファイルが見つかりません:"
    echo "   $JA_MODEL または $EN_MODEL"
    exit 1
fi

echo ""
echo "🚀 アプリを起動中..."
echo "   ブラウザで http://localhost:7862 を開いてください"
echo ""

python app_batch.py
