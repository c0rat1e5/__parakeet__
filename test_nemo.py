#!/usr/bin/env python3
"""CLIでNeMo直接テスト"""
import sys
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import ASRModel
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SCRIPT_DIR / "parakeet-tdt_ctc-0.6b-ja.nemo"

print(f"Loading model...")
model = ASRModel.restore_from(str(MODEL_PATH))
model.eval()
model = model.cuda()

# デコーディング設定を変更してCUDA Graphsを回避
# CTCモードを使用（TDTではなく）
print("Switching to CTC decoding mode...")
model.cur_decoder = "ctc"
print(f"Model loaded on GPU (using CTC decoder)")

# テスト用の短い音声ファイルがあれば指定
test_file = sys.argv[1] if len(sys.argv) > 1 else None

if test_file:
    print(f"Transcribing: {test_file}")
    output = model.transcribe([test_file])
    print(f"Result: {output}")
else:
    print("Usage: python test_nemo.py <audio_file>")
    print("No audio file provided, skipping transcription test")
