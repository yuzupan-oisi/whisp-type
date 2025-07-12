import os
import warnings

# OMP: Error #15 を回避するため
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ctranslate2から出るUserWarningを非表示にする
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

import pyautogui
from pynput import keyboard
import threading
import pyperclip
import time
import multiprocessing # Add this import
import sounddevice as sd

from config import app_config, load_config
from audio_processor import (
    audio_stream_generator,
    transcribe_audio,
    remove_filler_words,
    load_filler_words,
    SAMPLE_RATE,
    audio_callback,
    reset_recording_state, # reset_recording_state をインポート
)
from tray_menu import create_tray_icon

# --- グローバル変数 ---
last_transcribed_text = ""
_last_copied_text_by_app = None # アプリが最後にクリップボードにコピーしたテキスト
is_recording = threading.Event() # is_recordingをthreading.Eventに変更
transcription_thread = None
current_keys = set()
HOTKEY_COMBINATION = {keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.space}
transcription_history = [] 
MAX_PROMPT_CHARS = 200



def generate_initial_prompt():
    """文字起こし履歴からinitial_promptを生成する（文字数制限付き）"""
    default_prompt = app_config.get("default_initial_prompt", "")
    prompt_text = default_prompt
    current_len = len(prompt_text)

    # 履歴を古いものから順に結合し、MAX_PROMPT_CHARSを超えないようにする
    for text in transcription_history:
        separator = " " if not text.endswith(('。', '、', '.', ',')) else ""
        if current_len + len(separator) + len(text) <= MAX_PROMPT_CHARS:
            prompt_text += separator + text
            current_len += len(separator) + len(text)
        else:
            break
    return prompt_text

audio_input_queue = multiprocessing.Queue()
transcription_output_queue = multiprocessing.Queue()

def transcription_worker(input_queue, output_queue):
    """
    Whisperモデルをロードし、音声データを文字起こしするプロセス
    """
    print("Transcription worker process started.")
    while True:
        data = input_queue.get()
        if data is None:
            break
        audio_data, current_prompt = data
        
        original_text = transcribe_audio(audio_data, current_prompt=current_prompt)
        output_queue.put(original_text)
    print("Transcription worker process stopped.")


def insert_text_at_cursor(text):
    """現在のカーソル位置にテキストを挿入する"""
    global _last_copied_text_by_app
    print(f"Attempting to insert text: '{text}'")
    if text:
        _last_copied_text_by_app = text # アプリがコピーしたテキストを記録
        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'v')
        print(f"テキストを挿入しました: \"{text}\"")
        # 設定がTrueの場合のみクリップボードをクリア
        if app_config.get("clear_clipboard_after_insert", True):
            clear_clipboard_if_ours() 
    else:
        print("挿入するテキストがありません。")

def clear_clipboard_if_ours():
    """最後にアプリがコピーしたテキストであればクリップボードをクリアする"""
    global _last_copied_text_by_app
    try:
        current_clipboard_content = pyperclip.paste()
        if _last_copied_text_by_app and current_clipboard_content == _last_copied_text_by_app:
            pyperclip.copy("") # クリップボードを空にする
            print("クリップボードをクリアしました。")
        _last_copied_text_by_app = None # クリアしたかどうかにかかわらずリセット
    except pyperclip.PyperclipException as e:
        print(f"クリップボードの操作中にエラーが発生しました: {e}")

def transcription_loop():
    """
    音声ストリームを継続的に処理し、文字起こしとテキスト挿入を行うループ。
    is_recordingイベントがセットされている間、実行される。
    """
    global last_transcribed_text
    
    # audio_stream_generatorを開始
    stream = audio_stream_generator(is_recording)
    
    for audio_data in stream:
        if not is_recording.is_set():
            break

        duration_s = len(audio_data) / SAMPLE_RATE
        print(f"録音完了: {duration_s:.2f}秒間の音声データをキャプチャしました。")

        current_prompt = generate_initial_prompt()
        print(f"[DEBUG] Initial Prompt used: '{current_prompt}'")

        audio_input_queue.put((audio_data, current_prompt))
        original_text = transcription_output_queue.get()

        cleaned_text = remove_filler_words(original_text)

        transcription_history.append(cleaned_text)
        if len(transcription_history) > 5:
            transcription_history.pop(0)

        last_transcribed_text = cleaned_text

        print("----------------------------------------")
        print(f"[元テキスト]    : {original_text}")
        print(f"[フィラー除去後]: {cleaned_text}")
        print("----------------------------------------")

        insert_text_at_cursor(last_transcribed_text)

    print("文字起こしループを終了しました。")


def toggle_recording():
    """
    ホットキーによって呼び出され、録音の開始と停止を切り替える。
    """
    global transcription_thread
    if not is_recording.is_set():
        print("録音を開始します...")
        is_recording.set()
        transcription_thread = threading.Thread(target=transcription_loop, daemon=True)
        transcription_thread.start()
    else:
        print("録音を手動で停止します...")
        is_recording.clear()
        reset_recording_state() # 録音状態をリセット
        if transcription_thread:
            transcription_thread.join(timeout=2.0)
        transcription_thread = None
        print("録音を停止しました。")


def on_press(key):
    """キーが押されたときの処理"""
    global current_keys
    current_keys.add(key)
    if all(k in current_keys for k in HOTKEY_COMBINATION):
        # 誤作動を防ぐため、一度キーセットをクリア
        current_keys.clear()
        toggle_recording()

def on_release(key):
    """キーが離されたときの処理"""
    global current_keys
    try:
        current_keys.remove(key)
    except KeyError:
        pass

def main():
    """アプリケーションのメイン関数"""
    # 設定とフィラー語をロード
    load_config()
    load_filler_words()

    # 文字起こしプロセスを開始
    transcription_process = multiprocessing.Process(target=transcription_worker, args=(audio_input_queue, transcription_output_queue))
    transcription_process.daemon = True
    transcription_process.start()

    # 音声入力ストリームを開始
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        callback=audio_callback,
        device=app_config["mic_device_index"]
    )
    stream.start()
    print("マイク入力ストリームを開始しました。")

    # ホットキーリスナーを別スレッドで開始
    listener = keyboard.Listener(on_press=on_press, on_release=on_release, daemon=True)
    listener.start()
    print("ホットキーリスナーを開始しました (Ctrl+Alt+Space で録音開始/停止)。")

    # システムトレイアイコンを作成して実行
    icon = create_tray_icon(listener)
    try:
        icon.run()
    except KeyboardInterrupt:
        print("プログラムを終了します。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        # ストリームを停止
        stream.stop()
        stream.close()
        print("マイク入力ストリームを停止しました。")
        # プロセスを終了させるためのシグナルを送信
        audio_input_queue.put(None)
        transcription_process.join()
        listener.stop()

if __name__ == '__main__':
    main()
