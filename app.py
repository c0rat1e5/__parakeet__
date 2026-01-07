#!/usr/bin/env python3
"""
🦜 Parakeet TDT-CTC 0.6B-ja Web UI
ローカルホストで動作する日本語音声書き起こしWebアプリ

使用方法:
    python app.py
    
ブラウザで http://localhost:7860 を開く
"""

from nemo.collections.asr.models import ASRModel
import torch
import gradio as gr
import gc
import shutil
from pathlib import Path
from pydub import AudioSegment
import numpy as np
import os
import tempfile
import csv
import datetime
import subprocess

# ========================================
# 設定
# ========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SCRIPT_DIR / "parakeet-tdt_ctc-0.6b-ja.nemo"
TEMP_DIR = SCRIPT_DIR / "temp"

# CUDAエラーを回避するためCPUを使用（安定性優先）
# GPUを使用したい場合は "cuda" に変更
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルをグローバルに読み込み
print(f"🦜 モデルを読み込み中: {MODEL_PATH}")
print(f"   デバイス: {device}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")

model = ASRModel.restore_from(str(MODEL_PATH))
model.eval()

# 起動時にモデルをGPUに移動してウォームアップ
if device == "cuda":
    model = model.to(device)
    # ウォームアップ：CUDAコンテキストを初期化
    with torch.no_grad():
        dummy = torch.zeros(1, 16000, device=device)
        del dummy
        torch.cuda.empty_cache()
    print(f"   モデルをGPUに移動しました")

print("✅ モデル読み込み完了\n")


def start_session(request: gr.Request):
    """セッション開始時に一時ディレクトリを作成"""
    session_hash = request.session_hash if request else "default"
    session_dir = TEMP_DIR / session_hash
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session started: {session_hash}")
    return session_dir.as_posix()


def end_session(request: gr.Request):
    """セッション終了時に一時ファイルを削除"""
    session_hash = request.session_hash if request else "default"
    session_dir = TEMP_DIR / session_hash
    
    if session_dir.exists():
        shutil.rmtree(session_dir)
    
    print(f"Session ended: {session_hash}")


def extract_audio_from_video(video_path: str, output_path: str) -> bool:
    """動画ファイルから音声を抽出（16kHz mono WAV）"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False


def get_audio_segment(audio_path, start_second, end_second):
    """音声の一部を切り出し"""
    if not audio_path or not Path(audio_path).exists():
        return None
    try:
        start_ms = int(start_second * 1000)
        end_ms = int(end_second * 1000)
        start_ms = max(0, start_ms)
        if end_ms <= start_ms:
            end_ms = start_ms + 100

        audio = AudioSegment.from_file(audio_path)
        clipped_audio = audio[start_ms:end_ms]
        
        samples = np.array(clipped_audio.get_array_of_samples())
        if clipped_audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1).astype(samples.dtype)
        
        frame_rate = clipped_audio.frame_rate
        if frame_rate <= 0:
            frame_rate = audio.frame_rate
        
        if samples.size == 0:
            return None
        
        return (frame_rate, samples)
    except Exception as e:
        print(f"Error clipping audio: {e}")
        return None


def format_srt_time(seconds: float) -> str:
    """秒数をSRT形式のタイムスタンプに変換"""
    sanitized_total_seconds = max(0.0, seconds)
    delta = datetime.timedelta(seconds=sanitized_total_seconds)
    total_int_seconds = int(delta.total_seconds())
    
    hours = total_int_seconds // 3600
    remainder_seconds_after_hours = total_int_seconds % 3600
    minutes = remainder_seconds_after_hours // 60
    seconds_part = remainder_seconds_after_hours % 60
    milliseconds = delta.microseconds // 1000
    
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"


def generate_srt_content(segment_timestamps: list) -> str:
    """SRT形式のコンテンツを生成"""
    srt_content = []
    for i, ts in enumerate(segment_timestamps):
        start_time = format_srt_time(ts['start'])
        end_time = format_srt_time(ts['end'])
        text = ts['segment']
        srt_content.append(str(i + 1))
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")
    return "\n".join(srt_content)


def transcribe_audio(audio_input, session_dir):
    """音声ファイルを書き起こし"""
    # gr.Fileからのパス取得（ファイルオブジェクトまたはパス文字列に対応）
    if audio_input is None:
        print("エラー: 音声ファイルが指定されていません")
        return (
            [], [], None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            ""
        )
    
    # gr.Fileの場合はパスを取得、gr.Audioの場合はそのまま
    if hasattr(audio_input, 'name'):
        audio_path = audio_input.name
    elif isinstance(audio_input, str):
        audio_path = audio_input
    else:
        audio_path = str(audio_input)
    
    vis_data = [["N/A", "N/A", "処理失敗"]]
    raw_times_data = [[0.0, 0.0]]
    processed_audio_path = None
    csv_file_path = None
    srt_file_path = None
    txt_file_path = None
    full_text = ""
    
    original_path_name = Path(audio_path).name
    audio_name = Path(audio_path).stem
    
    # ボタンの初期状態（gr.update()を使用）
    csv_button = gr.update(visible=False)
    srt_button = gr.update(visible=False)
    txt_button = gr.update(visible=False)
    
    long_audio_settings_applied = False
    
    try:
        # 動画ファイルの場合は音声を抽出
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        if Path(audio_path).suffix.lower() in video_extensions:
            print(f"🎬 動画から音声を抽出中: {original_path_name}")
            temp_audio_path = Path(session_dir) / f"{audio_name}_extracted.wav"
            if not extract_audio_from_video(audio_path, str(temp_audio_path)):
                print("エラー: 動画からの音声抽出に失敗しました")
                return vis_data, raw_times_data, audio_path, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
            audio_path = str(temp_audio_path)
            original_path_name = temp_audio_path.name
        
        # 音声ファイルを読み込み
        print(f"🍜 音声を読み込み中: {original_path_name}")
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds
        
        # 16kHz モノラルに変換
        resampled = False
        mono = False
        
        target_sr = 16000
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
            resampled = True
        
        if audio.channels == 2:
            audio = audio.set_channels(1)
            mono = True
        elif audio.channels > 2:
            print(f"エラー: 音声が{audio.channels}チャンネルです")
            return vis_data, raw_times_data, audio_path, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
        
        if resampled or mono:
            processed_audio_path = Path(session_dir) / f"{audio_name}_processed.wav"
            audio.export(processed_audio_path, format="wav")
            transcribe_path = processed_audio_path.as_posix()
        else:
            transcribe_path = audio_path
        
        print(f"📝 書き起こし中... ({duration_sec:.1f}秒)")
        
        # 長い音声の場合は最適化設定を適用
        if duration_sec > 480:  # 8分以上
            try:
                print("⚡ 長い音声のため最適化設定を適用中...")
                model.change_attention_model("rel_pos_local_attn", [256, 256])
                model.change_subsampling_conv_chunking_factor(1)
                long_audio_settings_applied = True
            except Exception as e:
                print(f"Warning: Failed to apply long audio settings: {e}")
        
        # 推論
        output = model.transcribe([transcribe_path], timestamps=True)
        
        if not output or not isinstance(output, list) or not output[0]:
            print("エラー: 書き起こしに失敗しました")
            return vis_data, raw_times_data, audio_path, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
        
        # タイムスタンプを取得
        if hasattr(output[0], 'timestamp') and output[0].timestamp and 'segment' in output[0].timestamp:
            segment_timestamps = output[0].timestamp['segment']
        else:
            # タイムスタンプがない場合はテキスト全体を1セグメントとして扱う
            text = output[0].text if hasattr(output[0], 'text') else str(output[0])
            segment_timestamps = [{'start': 0.0, 'end': duration_sec, 'segment': text}]
        
        print(f"セグメント数: {len(segment_timestamps)}")
        
        # データ整形
        csv_headers = ["開始 (秒)", "終了 (秒)", "テキスト"]
        vis_data = [[f"{ts['start']:.2f}", f"{ts['end']:.2f}", ts['segment']] for ts in segment_timestamps]
        raw_times_data = [[ts['start'], ts['end']] for ts in segment_timestamps]
        
        # フルテキスト
        full_text = "".join([ts['segment'] for ts in segment_timestamps])
        print(f"フルテキスト長: {len(full_text)} 文字")
        
        # CSVファイルを保存
        try:
            csv_file_path = Path(session_dir) / f"{audio_name}_transcript.csv"
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)
                writer.writerows(vis_data)
            csv_button = gr.update(value=str(csv_file_path), visible=True)
        except Exception as e:
            print(f"CSV error: {e}")
        
        # SRTファイルを保存
        try:
            srt_content = generate_srt_content(segment_timestamps)
            srt_file_path = Path(session_dir) / f"{audio_name}_transcript.srt"
            with open(srt_file_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            srt_button = gr.update(value=str(srt_file_path), visible=True)
        except Exception as e:
            print(f"SRT error: {e}")
        
        # テキストファイルを保存
        try:
            txt_file_path = Path(session_dir) / f"{audio_name}_transcript.txt"
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            txt_button = gr.update(value=str(txt_file_path), visible=True)
        except Exception as e:
            print(f"TXT error: {e}")
        
        print("✅ 書き起こし完了！ 結果を返します...")
        
        # DataFrameの表示を最大500行に制限（ブラウザの負荷軽減）
        MAX_DISPLAY_ROWS = 500
        if len(vis_data) > MAX_DISPLAY_ROWS:
            print(f"⚠️ セグメント数が多いため表示を{MAX_DISPLAY_ROWS}行に制限します（全{len(vis_data)}行はCSVでダウンロード可能）")
            display_vis_data = vis_data[:MAX_DISPLAY_ROWS]
            display_raw_times = raw_times_data[:MAX_DISPLAY_ROWS]
        else:
            display_vis_data = vis_data
            display_raw_times = raw_times_data
        
        return display_vis_data, display_raw_times, audio_path, csv_button, srt_button, txt_button, full_text
    
    except torch.cuda.OutOfMemoryError:
        error_msg = 'GPUメモリ不足です。より短い音声で試してください。'
        print(f"OOM Error: {error_msg}")
        return [["OOM", "OOM", error_msg]], [[0.0, 0.0]], audio_path, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
    
    except Exception as e:
        error_msg = f"エラー: {e}"
        print(f"Transcription error: {e}")
        return [["Error", "Error", error_msg]], [[0.0, 0.0]], audio_path, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
    
    finally:
        # クリーンアップ
        try:
            if long_audio_settings_applied:
                model.change_attention_model("rel_pos")
                model.change_subsampling_conv_chunking_factor(-1)
            
            # GPUメモリをクリアするが、モデルはGPUに保持
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        # 処理済み音声を削除
        if processed_audio_path and os.path.exists(processed_audio_path):
            try:
                os.remove(processed_audio_path)
            except Exception:
                pass


def play_segment(evt: gr.SelectData, raw_ts_list, current_audio_path):
    """テーブルの行をクリックしたときにそのセグメントを再生"""
    if not isinstance(raw_ts_list, list) or not current_audio_path:
        return gr.Audio(value=None, label="選択したセグメント")
    
    selected_index = evt.index[0]
    
    if selected_index < 0 or selected_index >= len(raw_ts_list):
        return gr.Audio(value=None, label="選択したセグメント")
    
    if not isinstance(raw_ts_list[selected_index], (list, tuple)) or len(raw_ts_list[selected_index]) != 2:
        return gr.Audio(value=None, label="選択したセグメント")
    
    start_time_s, end_time_s = raw_ts_list[selected_index]
    segment_data = get_audio_segment(current_audio_path, start_time_s, end_time_s)
    
    if segment_data:
        return gr.Audio(
            value=segment_data, 
            autoplay=True, 
            label=f"セグメント: {start_time_s:.2f}秒 - {end_time_s:.2f}秒",
            interactive=False
        )
    else:
        return gr.Audio(value=None, label="選択したセグメント")


# ========================================
# Gradio UI
# ========================================
css = """
.main-title {
    text-align: center;
    margin-bottom: 1rem;
}
.info-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 1rem;
}
"""

with gr.Blocks(
    title="🦜 Parakeet 日本語音声書き起こし",
    css=css,
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🦜 Parakeet TDT-CTC 0.6B-ja
    ## 日本語音声書き起こしツール
    
    動画/音声ファイルをアップロードして、日本語テキストに書き起こします。
    
    **対応形式**: MP4, MKV, AVI, MOV, WebM, WAV, MP3, FLAC, OGG, M4A
    """)
    
    # 状態管理
    current_audio_path_state = gr.State(None)
    raw_timestamps_list_state = gr.State([])
    session_dir = gr.State()
    
    demo.load(start_session, outputs=[session_dir])
    
    with gr.Tabs():
        with gr.TabItem("📁 ファイルアップロード"):
            file_input = gr.File(
                label="音声/動画ファイルをアップロード",
                file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", 
                           ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
            )
            file_transcribe_btn = gr.Button("🎙️ 書き起こし開始", variant="primary", size="lg")
        
        with gr.TabItem("🎤 マイク録音"):
            mic_input = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="マイクで録音"
            )
            mic_transcribe_btn = gr.Button("🎙️ 書き起こし開始", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("### 📄 書き起こし結果")
    
    # ダウンロードボタン
    with gr.Row():
        download_btn_txt = gr.DownloadButton(label="📥 テキスト", visible=False)
        download_btn_srt = gr.DownloadButton(label="📥 SRT字幕", visible=False)
        download_btn_csv = gr.DownloadButton(label="📥 CSV", visible=False)
    
    # フルテキスト表示
    full_text_output = gr.Textbox(
        label="書き起こしテキスト（コピー可能）",
        lines=5,
        max_lines=10
    )
    
    # タイムスタンプ付きテーブル
    gr.Markdown("### ⏱️ タイムスタンプ付きセグメント（クリックで再生）")
    
    vis_timestamps_df = gr.DataFrame(
        headers=["開始 (秒)", "終了 (秒)", "テキスト"],
        datatype=["str", "str", "str"],
        wrap=True,
        label="セグメント一覧"
    )
    
    # セグメント再生プレイヤー
    selected_segment_player = gr.Audio(label="選択したセグメント", interactive=False)
    
    # イベントハンドラ
    file_transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[file_input, session_dir],
        outputs=[
            vis_timestamps_df, 
            raw_timestamps_list_state, 
            current_audio_path_state, 
            download_btn_csv, 
            download_btn_srt, 
            download_btn_txt,
            full_text_output
        ]
    )
    
    mic_transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[mic_input, session_dir],
        outputs=[
            vis_timestamps_df, 
            raw_timestamps_list_state, 
            current_audio_path_state, 
            download_btn_csv, 
            download_btn_srt, 
            download_btn_txt,
            full_text_output
        ]
    )
    
    vis_timestamps_df.select(
        fn=play_segment,
        inputs=[raw_timestamps_list_state, current_audio_path_state],
        outputs=[selected_segment_player]
    )
    
    demo.unload(end_session)
    
    gr.Markdown("""
    ---
    ### 📋 使い方
    1. **ファイルアップロード** または **マイク録音** タブを選択
    2. 音声/動画ファイルをアップロード、または録音
    3. 「書き起こし開始」ボタンをクリック
    4. 結果をテキスト、SRT字幕、またはCSVでダウンロード
    5. テーブルの行をクリックすると、そのセグメントを再生できます
    
    ### ⚙️ 技術情報
    - **モデル**: NVIDIA Parakeet TDT-CTC 0.6B-ja
    - **デバイス**: """ + device + """
    - **対応**: 日本語音声認識、句読点自動挿入
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("🦜 Parakeet 日本語音声書き起こし Web UI")
    print("=" * 60)
    print(f"\n📂 モデル: {MODEL_PATH}")
    print(f"💻 デバイス: {device}")
    print("\n🌐 ブラウザで http://localhost:7860 を開いてください\n")
    
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
