#!/usr/bin/env python3
"""
🦜 Speech-to-Text Batch Processor (日本語/English)
複数ファイルの一括書き起こしツール（WSL2対応版）

※ WSL2ではCTCデコーダーを使用（CUDA Graphsを回避）
   詳細は WSL2_CUDA_ERROR_README.md を参照
"""

from nemo.collections.asr.models import ASRModel
import torch
import gradio as gr
import gc
from pathlib import Path
from pydub import AudioSegment
import subprocess
import datetime
import zipfile
import tempfile

# ========================================
# 設定
# ========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
TEMP_DIR = SCRIPT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# 利用可能なモデル
MODELS = {
    "日本語 (Parakeet TDT-CTC 0.6B-ja)": {
        "path": SCRIPT_DIR / "parakeet-tdt_ctc-0.6b-ja.nemo",
        "lang": "ja",
    },
    "English (Nemotron Speech 0.6B-en)": {
        "path": SCRIPT_DIR / "nemotron-speech-streaming-en-0.6b.nemo",
        "lang": "en",
    },
}

# GPUを使用
device = "cuda" if torch.cuda.is_available() else "cpu"

# 現在のモデル（グローバル）
current_model = None
current_model_name = None


def load_model(model_name: str):
    """モデルを読み込み（または切り替え）"""
    global current_model, current_model_name

    if current_model_name == model_name and current_model is not None:
        return True

    model_info = MODELS.get(model_name)
    if not model_info or not model_info["path"].exists():
        print(
            f"❌ モデルファイルが見つかりません: {model_info['path'] if model_info else model_name}"
        )
        return False

    print(f"\n🦜 モデルを読み込み中: {model_name}")
    print(f"   パス: {model_info['path']}")
    print(f"   デバイス: {device}")

    # 古いモデルを解放
    if current_model is not None:
        del current_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # 新しいモデルを読み込み
    current_model = ASRModel.restore_from(str(model_info["path"]))
    current_model.eval()

    if device == "cuda":
        current_model = current_model.cuda()
        current_model.cur_decoder = "ctc"
        print("   モデルをGPUに移動しました (CTCデコーダー使用)")
    else:
        print("   CPUモードで動作中")

    current_model_name = model_name
    print("✅ モデル読み込み完了\n")
    return True


def extract_audio_from_video(video_path: str, output_path: str) -> bool:
    """動画ファイルから音声を抽出"""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False


def format_srt_time(seconds: float) -> str:
    """秒数をSRT形式のタイムスタンプに変換"""
    delta = datetime.timedelta(seconds=max(0.0, seconds))
    total_int_seconds = int(delta.total_seconds())
    hours = total_int_seconds // 3600
    minutes = (total_int_seconds % 3600) // 60
    seconds_part = total_int_seconds % 60
    milliseconds = delta.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"


def transcribe_single_file(audio_path: str, audio_name: str) -> tuple:
    """
    単一ファイルを書き起こし

    Returns:
        (segments, error_message)
    """
    MAX_CHUNK_SEC = 300  # 5分

    try:
        # 動画の場合は音声抽出
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        if Path(audio_path).suffix.lower() in video_extensions:
            print(f"   🎬 動画から音声を抽出中...")
            temp_audio = TEMP_DIR / f"{audio_name}_extracted.wav"
            if not extract_audio_from_video(audio_path, str(temp_audio)):
                return None, "動画からの音声抽出に失敗"
            audio_path = str(temp_audio)

        # 音声読み込み・変換
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds

        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        if audio.channels != 1:
            audio = audio.set_channels(1)

        processed_path = TEMP_DIR / f"{audio_name}_processed.wav"
        audio.export(processed_path, format="wav")

        # 書き起こし
        all_segments = []

        if duration_sec > MAX_CHUNK_SEC:
            # 長い音声: チャンク分割処理
            chunk_start = 0
            chunk_idx = 0

            while chunk_start < duration_sec:
                chunk_end = min(chunk_start + MAX_CHUNK_SEC, duration_sec)

                chunk_audio = audio[int(chunk_start * 1000) : int(chunk_end * 1000)]
                chunk_path = TEMP_DIR / f"{audio_name}_chunk_{chunk_idx}.wav"
                chunk_audio.export(chunk_path, format="wav")

                try:
                    output = current_model.transcribe(
                        [str(chunk_path)], timestamps=True
                    )

                    if output and output[0]:
                        if (
                            hasattr(output[0], "timestamp")
                            and output[0].timestamp
                            and "segment" in output[0].timestamp
                        ):
                            chunk_segments = output[0].timestamp["segment"]
                            for seg in chunk_segments:
                                seg["start"] += chunk_start
                                seg["end"] += chunk_start
                            all_segments.extend(chunk_segments)
                        else:
                            text = (
                                output[0].text
                                if hasattr(output[0], "text")
                                else str(output[0])
                            )
                            all_segments.append(
                                {
                                    "start": chunk_start,
                                    "end": chunk_end,
                                    "segment": text,
                                }
                            )
                finally:
                    if chunk_path.exists():
                        chunk_path.unlink()
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

                chunk_start = chunk_end
                chunk_idx += 1
        else:
            # 短い音声: そのまま処理
            output = current_model.transcribe([str(processed_path)], timestamps=True)

            if output and output[0]:
                if (
                    hasattr(output[0], "timestamp")
                    and output[0].timestamp
                    and "segment" in output[0].timestamp
                ):
                    all_segments = output[0].timestamp["segment"]
                else:
                    text = (
                        output[0].text if hasattr(output[0], "text") else str(output[0])
                    )
                    all_segments = [
                        {"start": 0.0, "end": duration_sec, "segment": text}
                    ]

        # クリーンアップ
        if processed_path.exists():
            processed_path.unlink()

        return all_segments, None

    except Exception as e:
        return None, str(e)


def save_transcript_files(segments: list, audio_name: str, output_dir: Path) -> tuple:
    """書き起こし結果をファイルに保存"""
    full_text = "".join([s["segment"] for s in segments])

    # テキストファイル
    txt_path = output_dir / f"{audio_name}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # SRTファイル
    srt_path = output_dir / f"{audio_name}_transcript.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments):
            f.write(f"{i+1}\n")
            f.write(f"{format_srt_time(s['start'])} --> {format_srt_time(s['end'])}\n")
            f.write(f"{s['segment']}\n\n")

    # CSVファイル
    csv_path = output_dir / f"{audio_name}_transcript.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Start,End,Text\n")
        for s in segments:
            f.write(f"{s['start']:.2f},{s['end']:.2f},\"{s['segment']}\"\n")

    return txt_path, srt_path, csv_path


def batch_transcribe(files, model_name, progress=gr.Progress()):
    """
    複数ファイルを一括書き起こし
    """
    if not files:
        return "⚠️ ファイルを選択してください", None, ""

    # モデル読み込み
    if not load_model(model_name):
        return f"❌ モデルの読み込みに失敗しました: {model_name}", None, ""

    # 出力ディレクトリ
    batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = TEMP_DIR / f"batch_{batch_id}"
    output_dir.mkdir(exist_ok=True)

    results = []
    errors = []
    total_files = len(files)

    print(f"\n{'='*60}")
    print(f"🦜 バッチ処理開始: {total_files}ファイル")
    print(f"   モデル: {model_name}")
    print(f"{'='*60}\n")

    for idx, file in enumerate(files):
        file_path = file.name if hasattr(file, "name") else str(file)
        file_name = Path(file_path).stem

        progress(
            (idx + 1) / total_files, f"処理中: {file_name} ({idx + 1}/{total_files})"
        )
        print(f"📝 [{idx + 1}/{total_files}] {file_name}")

        segments, error = transcribe_single_file(file_path, file_name)

        if error:
            errors.append(f"{file_name}: {error}")
            print(f"   ❌ エラー: {error}")
        elif segments:
            txt_path, srt_path, csv_path = save_transcript_files(
                segments, file_name, output_dir
            )
            full_text = "".join([s["segment"] for s in segments])
            results.append(
                {
                    "file": file_name,
                    "text": full_text,
                    "segments": len(segments),
                    "chars": len(full_text),
                }
            )
            print(f"   ✅ 完了: {len(segments)}セグメント, {len(full_text)}文字")
        else:
            errors.append(f"{file_name}: 書き起こし結果が空です")
            print(f"   ⚠️ 結果が空")

    # ZIPファイル作成
    zip_path = TEMP_DIR / f"batch_{batch_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.glob("*"):
            zf.write(file_path, file_path.name)

    # サマリー作成
    summary_lines = [
        f"# バッチ処理結果",
        f"",
        f"- **処理日時**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **モデル**: {model_name}",
        f"- **処理ファイル数**: {len(results)}/{total_files}",
        f"",
    ]

    if results:
        summary_lines.append("## ✅ 成功")
        for r in results:
            summary_lines.append(
                f"- **{r['file']}**: {r['segments']}セグメント, {r['chars']}文字"
            )

    if errors:
        summary_lines.append("")
        summary_lines.append("## ❌ エラー")
        for e in errors:
            summary_lines.append(f"- {e}")

    summary = "\n".join(summary_lines)

    # 詳細結果（各ファイルのテキスト）
    detail_lines = []
    for r in results:
        detail_lines.append(f"{'='*60}")
        detail_lines.append(f"📄 {r['file']}")
        detail_lines.append(f"{'='*60}")
        detail_lines.append(r["text"])
        detail_lines.append("")

    detail_text = "\n".join(detail_lines)

    print(f"\n{'='*60}")
    print(f"🎉 バッチ処理完了: {len(results)}/{total_files}ファイル成功")
    print(f"{'='*60}\n")

    return summary, str(zip_path), detail_text


# ========================================
# Gradio UI
# ========================================
with gr.Blocks(title="🦜 Batch Speech-to-Text") as demo:

    gr.Markdown(
        """
    # 🦜 Batch Speech-to-Text
    ## 複数ファイル一括書き起こし（日本語/English）
    
    複数の動画/音声ファイルを一括で書き起こします。
    """
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODELS.keys()),
            value=list(MODELS.keys())[0],
            label="🔄 言語/モデル選択",
            info="書き起こしに使用するモデルを選択",
        )

    file_input = gr.File(
        label="📁 ファイルをアップロード（複数選択可）",
        file_count="multiple",
        file_types=[
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".webm",
            ".wav",
            ".mp3",
            ".flac",
            ".m4a",
        ],
    )

    transcribe_btn = gr.Button("🚀 一括書き起こし開始", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### 📊 処理結果")

    summary_output = gr.Markdown(label="サマリー")

    zip_output = gr.File(label="📦 結果ダウンロード (ZIP)")

    with gr.Accordion("📄 詳細結果（各ファイルのテキスト）", open=False):
        detail_output = gr.Textbox(label="全テキスト", lines=20)

    transcribe_btn.click(
        fn=batch_transcribe,
        inputs=[file_input, model_dropdown],
        outputs=[summary_output, zip_output, detail_output],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🦜 Batch Speech-to-Text (日本語/English)")
    print("=" * 60)
    print("\n🌐 ブラウザで http://localhost:7862 を開いてください\n")

    demo.launch(server_name="0.0.0.0", server_port=7862)
