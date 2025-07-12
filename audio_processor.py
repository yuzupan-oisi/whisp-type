import sounddevice as sd
import numpy as np
import queue
import time
import re
import os
import torch
from faster_whisper import WhisperModel
from config import app_config, update_config

# --- 定数 ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCK_DURATION_MS = 100
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)
SILENCE_THRESHOLD = 300

MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"


# --- グローバル変数 ---
audio_queue = queue.Queue()
model = None
FILLER_WORDS = []

def load_filler_words():
    global FILLER_WORDS
    filler_file = app_config["filler_words_file"]
    if os.path.exists(filler_file):
        with open(filler_file, "r", encoding="utf-8") as f:
            FILLER_WORDS = [line.strip() for line in f if line.strip()]
    else:
        default_filler_words = [
            "えーっと", "えーと", "えっと", "えー", "ええ",
            "あー", "あーあ", "ああ",
            "あのー", "あの",
            "うーん",
            "なんか",
            "まあ",
            "そのー", "その",
        ]
        with open(filler_file, "w", encoding="utf-8") as f:
            for word in default_filler_words:
                f.write(word + "\n")
        FILLER_WORDS = default_filler_words

def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())



recorded_frames = []
is_speaking = False
silence_start_time = None
waiting_start_time = None

def reset_recording_state():
    """録音状態をリセットする"""
    global recorded_frames, is_speaking, silence_start_time, waiting_start_time
    recorded_frames = []
    is_speaking = False
    silence_start_time = None
    waiting_start_time = None
    # キューをクリアする
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

def audio_stream_generator(is_recording):
    """
    音声入力ストリームを管理し、無音区間で区切られた音声チャンクを生成するジェネレータ。
    `is_recording`がFalseになるか、一定時間（long_silence_duration_s）待機状態が続いた場合に停止する。
    """
    global recorded_frames, is_speaking, silence_start_time, waiting_start_time

    reset_recording_state() # 開始時に状態をリセット
    waiting_start_time = time.time()
    long_silence_duration = app_config.get("long_silence_duration_s", 10.0)
    print("\nマイクに向かって話してください。待機中...")

    while is_recording.is_set():
        # --- 1. 長い無音による自動停止をチェック (ループの最優先事項) ---
        if waiting_start_time:
            elapsed_time = time.time() - waiting_start_time
            if elapsed_time > long_silence_duration:
                print(f"{long_silence_duration}秒間待機状態が続いたため、録音を自動停止します。")
                is_recording.clear()
                break

        # --- 2. キューから音声データを取得 ---
        try:
            audio_chunk = audio_queue.get(timeout=0.1) # タイムアウト付きで待機
        except queue.Empty:
            continue # タイムアウトした場合はループの先頭に戻り、is_recordingを再チェック

        # --- 3. 音声データを処理 ---
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))

        if rms > SILENCE_THRESHOLD:
            if not is_speaking:
                print("音声検知...")
                is_speaking = True
                if waiting_start_time:
                    waiting_start_time = None
            
            recorded_frames.append(audio_chunk)
            silence_start_time = None

        elif is_speaking:
            recorded_frames.append(audio_chunk)
            if silence_start_time is None:
                silence_start_time = time.time()

            if time.time() - silence_start_time > app_config.get("silence_duration_s", 2.0):
                print(f"{app_config.get('silence_duration_s', 2.0)}秒間の無音を検出。音声チャンクを処理します。")
                if recorded_frames:
                    audio_data = np.concatenate(recorded_frames, axis=0)
                    yield audio_data

                is_speaking = False
                silence_start_time = None
                recorded_frames = []
                if waiting_start_time is None:
                    waiting_start_time = time.time()
                print("\nマイクに向かって話してください。待機中...")

    if recorded_frames:
        print("録音終了。残りの音声チャンクを処理します。")
        audio_data = np.concatenate(recorded_frames, axis=0)
        yield audio_data
    
    reset_recording_state() # 終了時にも状態をリセット



def remove_filler_words(text):

    filler_pattern = "|".join(FILLER_WORDS)
    cleaned_text = re.sub(filler_pattern, "", text)
    cleaned_text = re.sub(r"([、。,\s])\1+", r"\1", cleaned_text).strip()
    cleaned_text = re.sub(r"^[、。,\s]+", "", cleaned_text)
    return cleaned_text

def transcribe_audio(audio_data, current_prompt=""):
    global model
    if model is None:
        device = "cpu"
        compute_type = app_config.get("compute_type", "int8")

        if app_config.get("use_gpu", False):
            if torch.cuda.is_available():
                device = "cuda"
                print(f"CUDA (GPU) が利用可能です。device='{device}', compute_type='{compute_type}' でモデルをロードします。")
            else:
                print("警告: 'use_gpu'がTrueですが、CUDAが利用できません。CPUにフォールバックします。")
                # use_gpu設定は変更せず、このセッションのみCPUを使用する
        else:
            print(f"CPUを使用します。device='{device}', compute_type='{compute_type}' でモデルをロードします。")

        print(f"Whisperモデル({MODEL_SIZE})をロードしています...")
        try:
            model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
            print("モデルのロードが完了しました。")
        except Exception as e:
            print(f"エラー: モデルのロードに失敗しました。 {e}")
            if device == "cuda":
                print("CUDAでのモデルロードに失敗したため、CPUにフォールバックして再試行します。")
                device = "cpu"
                model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
                print("CPUでのモデルロードが完了しました。")
            else:
                raise e

    if audio_data is None:
        return ""

    audio_flat = audio_data.flatten()
    audio_float32 = audio_flat.astype(np.float32) / 32768.0

    print("文字起こしを開始します...")
    start_time = time.time()
    segments, info = model.transcribe(
        audio_float32,
        beam_size=5,
        language=app_config["language"],
        vad_filter=True,
        initial_prompt=current_prompt
    )

    print(f"[DEBUG] Detected language: '{info.language}' with probability {info.language_probability:.2f}")
    transcribed_text = "".join(segment.text for segment in segments)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"文字起こし完了 (処理時間: {processing_time:.2f}秒)")

    return transcribed_text

# --- 初期化 ---
load_filler_words()
