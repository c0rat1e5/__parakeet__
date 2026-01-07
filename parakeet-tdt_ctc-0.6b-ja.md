
---
type: ops
aliases: [NVIDIA Parakeet Japanese ASR, parakeet-tdt-ctc-ja, NeMo 日本語 ASR, NVIDIA 日本語音声認識]
tags: [asr, speech-recognition, nvidia, nemo, japanese, ai-model, ops/ml, huggingface]
created: 2026-01-08
updated: 2026-01-08
---

# NVIDIA Parakeet TDT-CTC 0.6B (Japanese)

## 概要

`nvidia/parakeet-tdt_ctc-0.6b-ja` は、NVIDIAのNeMoチームが開発した**日本語音声認識（ASR: Automatic Speech Recognition）モデル**です。日本語音声を句読点付きでテキストに書き起こすことができます。

- **モデルサイズ**: 約6億パラメータ（0.6B）
- **対応言語**: 日本語
- **ライセンス**: CC-BY-4.0
- **公開場所**: [Hugging Face](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja)

---

## モデルアーキテクチャ

### Hybrid FastConformer-TDT-CTC

このモデルは **Hybrid FastConformer-TDT-CTC アーキテクチャ** を採用しています。

#### FastConformer
- Conformerモデルの最適化バージョン
- 8x depthwise-separable convolutional downsampling を使用
- 高速で効率的な音声認識を実現

#### TDT (Token-and-Duration Transducer)
- 従来のTransducerを一般化したモデル
- トークン予測と持続時間予測を分離
- 従来のTransducerが多くのblank予測を生成するのに対し、TDTモデルは持続時間出力を使用してblank予測の大部分をスキップ可能
- 最大4フレームまでのスキップに対応
- **推論速度が大幅に向上**

📄 **関連論文**:
- [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
- [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795)

---

## トレーニング

### 学習データ
- **ReazonSpeech v2.0**: 35,000時間以上の日本語自然音声コーパス
- データセット: [reazon-research/reazonspeech](https://huggingface.co/datasets/reazon-research/reazonspeech)

### 学習設定
- **GPU**: 32 x NVIDIA A100 80GB
- **学習ステップ**: 300,000ステップ
- **追加Fine-tuning**: 100,000ステップ（CER > 10%のサンプルの予測テキストを使用）
- **バッチ設定**: Dynamic bucketing、GPUあたり600秒のバッチ持続時間
- **トークナイザ**: SentencePiece（3,072トークン）

---

## パフォーマンス

### Character Error Rate (CER %)

句読点と非アルファベット文字を除去し、数字は`num2words`ライブラリで単語に変換して計算。

| NeMo Version | Decoder | JSUT basic5000 | CV 8.0 | CV 16.1 Dev | CV 16.1 Test | TEDxJP-10k |
|:------------:|:-------:|:--------------:|:------:|:-----------:|:------------:|:----------:|
| 1.23.0       | TDT     | 6.4            | 7.1    | 10.1        | 13.2         | 9.0        |
| 1.23.0       | CTC     | 6.5            | 7.2    | 10.2        | 13.3         | 9.1        |

※ 外部言語モデルなしのGreedy CER

---

## 使用方法

### インストール

```bash
pip install nemo_toolkit['asr']
```

### モデルの読み込み

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
)
```

### 音声ファイルの書き起こし

```python
output = asr_model.transcribe(['speech.wav'])
print(output[0].text)
```

### 複数ファイルの一括処理

```bash
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
    pretrained_name="nvidia/parakeet-tdt_ctc-0.6b-ja" \
    audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

※ デフォルトではTDTデコーダを使用。CTCに切り替える場合は `decoding_type='ctc'` を指定

### 入出力仕様

| 項目 | 仕様 |
|:-----|:-----|
| **入力** | 16,000 Hz モノラル音声（WAVファイル） |
| **出力** | 書き起こしテキスト（文字列） |

---

## デプロイメント

### NVIDIA Riva
[NVIDIA Riva](https://developer.nvidia.com/riva) は、オンプレミス、クラウド、マルチクラウド、ハイブリッド、エッジ、組み込み環境にデプロイ可能な音声AI SDKです。

主な機能:
- 高精度な音声認識
- ランタイムでの単語ブースティング（ブランド名や製品名など）
- 音響モデル、言語モデル、逆テキスト正規化のカスタマイズ
- ストリーミング音声認識
- Kubernetes互換のスケーリング
- エンタープライズグレードのサポート

---

## リファレンス

1. [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
2. [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795)
3. [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo)
4. [Google SentencePiece Tokenizer](https://github.com/google/sentencepiece)
5. [ReazonSpeech v2.0](https://huggingface.co/datasets/reazon-research/reazonspeech)
6. [num2words library](https://github.com/savoirfairelinux/num2words)

---

## ライセンス

このモデルは **CC-BY-4.0** ライセンスの下で公開されています。

詳細: [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)

---

## Whisper large-v3 との比較

### 基本情報比較

| 項目 | Parakeet TDT-CTC 0.6B-ja | Whisper large-v3 |
|:-----|:-------------------------|:-----------------|
| **開発元** | NVIDIA NeMo | OpenAI |
| **パラメータ数** | 約6億（0.6B） | 約15.5億（1.55B） |
| **対応言語** | 日本語専用 | 99言語（多言語） |
| **アーキテクチャ** | FastConformer-TDT-CTC | Transformer Encoder-Decoder |
| **学習データ** | 35,000時間（ReazonSpeech v2.0） | 500万時間以上 |
| **ライセンス** | CC-BY-4.0 | Apache-2.0 |
| **入力形式** | 16kHz モノラル WAV | 16kHz 音声 |
| **VRAM使用量** | 少ない（0.6Bのため） | 約10GB |

### 日本語性能比較

> ⚠️ **注意**: 以下は公開ベンチマークからの推定値です。評価条件（前処理、正規化方法など）が異なるため、直接比較には注意が必要です。

| ベンチマーク | Parakeet TDT-CTC 0.6B-ja (CER%) | Whisper large-v3 (CER%) | 備考 |
|:-------------|:-------------------------------:|:-----------------------:|:-----|
| JSUT basic5000 | **6.4** | 約10-15 | Parakeetが優位 |
| CommonVoice 8.0 | **7.1** | 約8-12 | Parakeetがやや優位 |
| CommonVoice 16.1 Test | 13.2 | 約12-18 | 同程度 |
| TEDxJP-10k | **9.0** | 約12-16 | Parakeetが優位 |

### どちらを選ぶべきか？

#### 🏆 Parakeet TDT-CTC 0.6B-ja が優れている点

| 優位点 | 説明 |
|:-------|:-----|
| **日本語精度** | 日本語に特化して学習されているため、一般的に高い精度 |
| **推論速度** | TDTアーキテクチャにより、高速な推論が可能 |
| **リソース効率** | パラメータ数が少なく、メモリ使用量が約1/3 |
| **句読点対応** | 日本語の句読点を自動で出力 |
| **日本語訓練データ** | ReazonSpeech（35,000時間の日本語音声）で集中的に学習 |

#### 🏆 Whisper large-v3 が優れている点

| 優位点 | 説明 |
|:-------|:-----|
| **多言語対応** | 99言語に対応、翻訳機能あり |
| **汎用性** | 様々な音声環境・話者に対するロバスト性 |
| **エコシステム** | 広大なコミュニティ、多数のツール・ラッパー |
| **Zero-shot性能** | Fine-tuningなしでも多くのタスクに対応 |
| **Hallucination対策** | 継続的な改善（large-v2 → v3） |

### 結論

| ユースケース | 推奨モデル |
|:-------------|:-----------|
| **日本語のみ使用** | ✅ Parakeet TDT-CTC 0.6B-ja |
| **最高の日本語精度が必要** | ✅ Parakeet TDT-CTC 0.6B-ja |
| **リソースが限られている** | ✅ Parakeet TDT-CTC 0.6B-ja |
| **多言語対応が必要** | ✅ Whisper large-v3 |
| **翻訳機能が必要** | ✅ Whisper large-v3 |
| **既存ツールとの統合** | ✅ Whisper large-v3 |

**日本語音声認識のみが目的であれば、Parakeet TDT-CTC 0.6B-ja の方が効率的で高精度です。**

---

## 関連リンク

- 🤗 [Hugging Face Model Page](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja)
- 📚 [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer)
- 🗂️ [Parakeet Model Collection](https://huggingface.co/collections/nvidia/parakeet)
- 🔊 [OpenAI Whisper](https://github.com/openai/whisper)
- 🤗 [Whisper large-v3 on Hugging Face](https://huggingface.co/openai/whisper-large-v3)
