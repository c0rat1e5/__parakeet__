#!/usr/bin/env python3
"""
🦜 Parakeet TDT-CTC 0.6B-ja Web UI
日本語音声書き起こしツール（WSL2対応版）

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

# ========================================
# 設定
# ========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SCRIPT_DIR / "parakeet-tdt_ctc-0.6b-ja.nemo"
TEMP_DIR = SCRIPT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# GPUを使用
device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルをグローバルに読み込み
print(f"🦜 モデルを読み込み中: {MODEL_PATH}")
print(f"   デバイス: {device}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")

model = ASRModel.restore_from(str(MODEL_PATH))
model.eval()

# GPUに移動
if device == "cuda":
    model = model.cuda()
    # WSL2互換性のためCTCデコーダーを使用（CUDA Graphsを回避）
    model.cur_decoder = "ctc"
    print("   モデルをGPUに移動しました (CTCデコーダー使用)")
else:
    print("   CPUモードで動作中")

print("✅ モデル読み込み完了\n")


def extract_audio_from_video(video_path: str, output_path: str) -> bool:
    """動画ファイルから音声を抽出"""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
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


def transcribe_short_audio(audio_path: str, duration_sec: float) -> list:
    """
    短い音声（5分以下）をそのまま書き起こし
    
    Args:
        audio_path: 音声ファイルのパス
        duration_sec: 音声の長さ（秒）
    
    Returns:
        セグメントのリスト [{'start': float, 'end': float, 'segment': str}, ...]
    """
    output = model.transcribe([audio_path], timestamps=True)
    
    if not output or not output[0]:
        return []
    
    if hasattr(output[0], 'timestamp') and output[0].timestamp and 'segment' in output[0].timestamp:
        return output[0].timestamp['segment']
    else:
        text = output[0].text if hasattr(output[0], 'text') else str(output[0])
        return [{'start': 0.0, 'end': duration_sec, 'segment': text}]


def transcribe_long_audio(audio: AudioSegment, audio_name: str, duration_sec: float) -> list:
    """
    長い音声を5分ごとに分割して書き起こし
    
    Args:
        audio: AudioSegmentオブジェクト
        audio_name: ファイル名（拡張子なし）
        duration_sec: 音声の長さ（秒）
    
    Returns:
        セグメントのリスト [{'start': float, 'end': float, 'segment': str}, ...]
    """
    MAX_CHUNK_SEC = 300  # 5分
    
    print(f"⚡ 長い音声のため{MAX_CHUNK_SEC}秒ごとに分割処理...")
    all_segments = []
    chunk_start = 0
    chunk_idx = 0
    
    while chunk_start < duration_sec:
        chunk_end = min(chunk_start + MAX_CHUNK_SEC, duration_sec)
        print(f"   チャンク {chunk_idx + 1}: {chunk_start:.0f}秒 - {chunk_end:.0f}秒")
        
        # チャンクを切り出し
        chunk_audio = audio[int(chunk_start * 1000):int(chunk_end * 1000)]
        chunk_path = TEMP_DIR / f"{audio_name}_chunk_{chunk_idx}.wav"
        chunk_audio.export(chunk_path, format="wav")
        
        try:
            # 書き起こし
            output = model.transcribe([str(chunk_path)], timestamps=True)
            
            if output and output[0]:
                if hasattr(output[0], 'timestamp') and output[0].timestamp and 'segment' in output[0].timestamp:
                    chunk_segments = output[0].timestamp['segment']
                    # タイムスタンプにオフセットを加算
                    for seg in chunk_segments:
                        seg['start'] += chunk_start
                        seg['end'] += chunk_start
                    all_segments.extend(chunk_segments)
                else:
                    text = output[0].text if hasattr(output[0], 'text') else str(output[0])
                    all_segments.append({'start': chunk_start, 'end': chunk_end, 'segment': text})
        finally:
            # チャンクファイルを削除
            if chunk_path.exists():
                chunk_path.unlink()
            # GPUメモリをクリア
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        
        chunk_start = chunk_end
        chunk_idx += 1
    
    return all_segments


def save_transcript_files(segments: list, audio_name: str) -> tuple:
    """
    書き起こし結果をファイルに保存
    
    Args:
        segments: セグメントのリスト
        audio_name: ファイル名（拡張子なし）
    
    Returns:
        (full_text, txt_path, srt_path, csv_path)
    """
    full_text = "".join([s['segment'] for s in segments])
    
    # テキストファイル
    txt_path = TEMP_DIR / f"{audio_name}_transcript.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    # SRTファイル
    srt_path = TEMP_DIR / f"{audio_name}_transcript.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, s in enumerate(segments):
            f.write(f"{i+1}\n")
            f.write(f"{format_srt_time(s['start'])} --> {format_srt_time(s['end'])}\n")
            f.write(f"{s['segment']}\n\n")
    
    # CSVファイル
    csv_path = TEMP_DIR / f"{audio_name}_transcript.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("開始,終了,テキスト\n")
        for s in segments:
            f.write(f"{s['start']:.2f},{s['end']:.2f},\"{s['segment']}\"\n")
    
    return full_text, str(txt_path), str(srt_path), str(csv_path)


def transcribe_audio(audio_input):
    """
    音声/動画ファイルを書き起こし（メイン処理）
    
    処理フロー:
    1. 動画の場合は音声を抽出
    2. 16kHz モノラルに変換
    3. 5分以上の場合はチャンク分割処理
    4. 結果をファイルに保存
    """
    if audio_input is None:
        return "⚠️ ファイルを選択してください", None, None, None
    
    # パス取得
    if hasattr(audio_input, 'name'):
        audio_path = audio_input.name
    else:
        audio_path = str(audio_input)
    
    audio_name = Path(audio_path).stem
    
    try:
        # ========================================
        # 1. 動画の場合は音声を抽出
        # ========================================
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        if Path(audio_path).suffix.lower() in video_extensions:
            print(f"🎬 動画から音声を抽出中...")
            temp_audio = TEMP_DIR / f"{audio_name}_extracted.wav"
            if not extract_audio_from_video(audio_path, str(temp_audio)):
                return "❌ 動画からの音声抽出に失敗しました", None, None, None
            audio_path = str(temp_audio)
        
        # ========================================
        # 2. 音声を16kHz モノラルに変換
        # ========================================
        print(f"🎵 音声を読み込み中...")
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds
        
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        if audio.channels != 1:
            audio = audio.set_channels(1)
        
        processed_path = TEMP_DIR / f"{audio_name}_processed.wav"
        audio.export(processed_path, format="wav")
        
        # ========================================
        # 3. 書き起こし（長さに応じて処理を分岐）
        # ========================================
        print(f"📝 書き起こし中... ({duration_sec:.1f}秒)")
        
        MAX_CHUNK_SEC = 300  # 5分
        
        if duration_sec > MAX_CHUNK_SEC:
            # 長い音声: チャンク分割処理
            segments = transcribe_long_audio(audio, audio_name, duration_sec)
        else:
            # 短い音声: そのまま処理
            segments = transcribe_short_audio(str(processed_path), duration_sec)
        
        if not segments:
            return "❌ 書き起こしに失敗しました", None, None, None
        
        print(f"✅ 完了！ {len(segments)}セグメント")
        
        # ========================================
        # 4. 結果をファイルに保存
        # ========================================
        full_text, txt_path, srt_path, csv_path = save_transcript_files(segments, audio_name)
        print(f"   {len(full_text)}文字")
        
        # クリーンアップ
        if processed_path.exists():
            processed_path.unlink()
        
        return full_text, txt_path, srt_path, csv_path
    
    except Exception as e:
        print(f"Error: {e}")
        return f"❌ エラー: {e}", None, None, None


# ========================================
# Gradio UI
# ========================================
with gr.Blocks(title="🦜 Parakeet 日本語音声書き起こし") as demo:
    
    gr.Markdown("""
    # 🦜 Parakeet TDT-CTC 0.6B-ja
    ## 日本語音声書き起こしツール
    """)
    
    file_input = gr.File(
        label="音声/動画ファイルをアップロード",
        file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".wav", ".mp3", ".flac", ".m4a"]
    )
    
    transcribe_btn = gr.Button("🎙️ 書き起こし開始", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("### 📄 結果")
    
    result_text = gr.Textbox(label="書き起こしテキスト", lines=10)
    
    with gr.Row():
        txt_file = gr.File(label="📥 テキスト")
        srt_file = gr.File(label="📥 SRT字幕")
        csv_file = gr.File(label="📥 CSV")
    
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[file_input],
        outputs=[result_text, txt_file, srt_file, csv_file]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🦜 Parakeet 日本語音声書き起こし Web UI (シンプル版)")
    print("=" * 60)
    
    demo.launch(server_name="0.0.0.0", server_port=7860)
