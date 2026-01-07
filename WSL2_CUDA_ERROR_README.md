# ⚠️ WSL2 CUDA Graphs エラーと解決策

## 🚨 重要：このファイルは必ず読んでください

WSL2環境でParakeet TDT-CTC モデルを使用する際、**CUDA failure 35** エラーが発生します。
このエラーの原因と解決策を記録しています。

---

## 💥 発生するエラー

```
Exception: CUDA failure! 35
```

または

```
CUDA driver error: out of memory
```

### エラーの詳細スタックトレース

```python
File ".../transducer_decoding/tdt_label_looping.py", line 1029, in _full_graph_compile
    capture_status, _, graph, _, _, _ = cu_call(
        cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
    )
File ".../cuda_python_utils.py", line 101, in cu_call
    raise Exception(f"CUDA failure! {error}")
Exception: CUDA failure! 35
```

---

## 🔍 原因

### CUDA error 35 = `CUDA_ERROR_INSUFFICIENT_DRIVER`

**NeMoのTDTデコーダーはCUDA Graphsを使用** しています。

CUDA Graphsとは：
- GPU操作を事前にキャプチャして高速実行する機能
- **WSL2のCUDAドライバとの互換性に問題がある**
- ネイティブLinuxやWindows直接実行では問題なし

### なぜWSL2で問題が起きるか

1. WSL2はWindowsのCUDAドライバを経由してGPUにアクセス
2. CUDA Graphsの `cudaStreamGetCaptureInfo` がWSL2環境で正しく動作しない
3. TDTデコーダーは `tdt_label_looping.py` でCUDA Graphsを必須として使用
4. → **CUDA failure 35** が発生

---

## ✅ 解決策：CTCデコーダーに切り替え

### 修正コード

```python
model = ASRModel.restore_from(str(MODEL_PATH))
model.eval()
model = model.cuda()

# ★★★ WSL2互換性のためCTCデコーダーを使用 ★★★
# TDTデコーダーはCUDA Graphsを使用するためWSL2で動作しない
model.cur_decoder = "ctc"
```

### なぜCTCデコーダーなら動くか

| デコーダー | CUDA Graphs | WSL2互換性 |
|-----------|-------------|-----------|
| **TDT** | 使用する | ❌ エラー発生 |
| **CTC** | 使用しない | ✅ 動作する |

Parakeet TDT-CTC モデルは **ハイブリッドモデル** で、TDTとCTCの両方のデコーダーを持っています。
`model.cur_decoder = "ctc"` で切り替えることでCUDA Graphsを回避できます。

---

## 📊 精度への影響

| 項目 | TDTデコーダー | CTCデコーダー |
|------|--------------|--------------|
| **精度** | より高い（本来の設計） | 若干低下の可能性 |
| **速度** | 高速（CUDA Graphs） | やや遅い |
| **WSL2** | ❌ 動作しない | ✅ 動作する |

**実用上は大きな差はありません。** WSL2環境ではCTCデコーダーを使用してください。

---

## 🔧 他に試した解決策（失敗）

### 1. CPUモードで実行
```python
device = "cpu"
```
→ **失敗**: 25分の動画でWSL自体がフリーズ（メモリ不足）

### 2. multiprocessing spawn設定
```python
multiprocessing.set_start_method('spawn', force=True)
```
→ **失敗**: CUDA Graphsの問題は解決せず

### 3. CUDA_VISIBLE_DEVICES=""
```bash
CUDA_VISIBLE_DEVICES="" python app.py
```
→ **失敗**: CPUモードと同じくメモリ不足

### 4. デコーディング設定の変更
```python
decoding_cfg.greedy.loop_labels = False
```
→ **失敗**: 設定キーが存在しない

---

## 📁 関連ファイル

- `app_simple.py` - WSL2対応版（CTCデコーダー使用）
- `app.py` - 元のバージョン（TDTデコーダー、WSL2で動作しない）
- `test_nemo.py` - デバッグ用テストスクリプト

---

## 🖥️ 動作確認環境

```
OS: WSL2 (Ubuntu on Windows)
GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16GB)
CUDA: 12.9
PyTorch: 2.9.1+cu128
NeMo: 最新版
Driver: 576.83 (Windows) / 575.65 (WSL2)
```

---

## 📅 記録日

2026年1月8日

---

## 🎯 結論

**WSL2環境では必ず `model.cur_decoder = "ctc"` を設定すること！**

```python
# これを忘れると CUDA failure! 35 が発生する
model.cur_decoder = "ctc"
```

ネイティブLinuxまたはWindows直接実行環境では、TDTデコーダー（デフォルト）を使用できます。
