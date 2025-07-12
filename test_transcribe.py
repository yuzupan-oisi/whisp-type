import librosa
from faster_whisper import WhisperModel
import time

# --- パラメータ ---
MODEL_SIZE = "small"  # 前回試したモデル
COMPUTE_TYPE = "int8"
AUDIO_FILE = "debug_audio.wav" # 正常に再生できた音声ファイル

def main():
    """
    音声ファイルを直接読み込んで文字起こしだけを行うテスト。
    """
    print("--- 文字起こし単体テスト --- ")

    # 1. Whisperモデルをロード
    print(f"Whisperモデル({MODEL_SIZE})をロードしています...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return
    print("モデルのロードが完了しました。")

    # 2. librosaで音声ファイルをロード
    # librosaは自動的にサンプリングレートを16kHzに変換し、モノラルにしてくれます。
    print(f"音声ファイル {AUDIO_FILE} をロードしています...")
    try:
        audio_data, sample_rate = librosa.load(AUDIO_FILE, sr=16000, mono=True)
    except Exception as e:
        print(f"音声ファイルのロード中にエラーが発生しました: {e}")
        print("'debug_audio.wav'が存在し、有効な音声ファイルであることを確認してください。")
        print("もしエラーが続く場合、音声コーデックの問題かもしれません。その場合はffmpegのインストールをお試しください。")
        return
    print(f"音声ファイルのロード完了。継続時間: {len(audio_data)/sample_rate:.2f}秒")

    # 3. 文字起こしを実行
    print("文字起こしを開始します...")
    start_time = time.time()

    try:
        segments, info = model.transcribe(
            audio_data,
            beam_size=5,
            language="ja",
            vad_filter=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8)
        )

        print(f"[DEBUG] Detected language: '{info.language}' with probability {info.language_probability:.2f}")
        transcribed_text = "".join(segment.text for segment in segments)

    except Exception as e:
        print(f"文字起こし中にエラーが発生しました: {e}")
        return

    end_time = time.time()
    print(f"文字起こし完了 (処理時間: {end_time - start_time:.2f}秒)")

    # 4. 結果を出力
    print("\n--- 結果 ---")
    print(transcribed_text)
    print("------------")

if __name__ == '__main__':
    main()
