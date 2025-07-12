import json
import os

# --- 定数 ---
CONFIG_FILE = "config.json"

# デフォルト設定
DEFAULT_CONFIG = {
    "language": "ja",
    "mic_device_index": None, # Noneはデフォルトのマイクを使用
    "filler_words_file": "filler_words.txt", # フィラー語リストのファイル名
    "use_gpu": True, # GPUを使用するかどうか
    "compute_type": "int8", # "int8" or "float16"
    "long_silence_duration_s": 3.0, # 長い沈黙の閾値（秒）
    "default_initial_prompt": "以下の内容について、実装を行います。", # 初期プロンプト
    "clear_clipboard_after_insert": False, # 挿入後にクリップボードをクリアするかどうか
    "silence_duration_s": 1.0 # 沈黙の閾値（秒）
}

# --- グローバル変数 ---
app_config = {}

def load_config():
    """設定ファイルを読み込む。ファイルが存在しない、または破損している場合はデフォルト設定を再生成する。"""
    global app_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            # JSONのトップレベルがオブジェクト(dict)であることを確認
            if not isinstance(config, dict):
                raise TypeError("設定ファイルが不正な形式です。")
            # 既存の設定に新しい設定項目が追加された場合に対応
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            app_config = config
            return
        except (json.JSONDecodeError, TypeError) as e:
            print(f"警告: '{CONFIG_FILE}' の読み込みに失敗しました ({e})。デフォルト設定で上書きします。")
            save_config(DEFAULT_CONFIG)
            app_config = DEFAULT_CONFIG.copy()
            return
    else:
        save_config(DEFAULT_CONFIG)
        app_config = DEFAULT_CONFIG.copy()
        return

def save_config(config):
    """設定ファイルを保存する。"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def update_config(key, value):
    global app_config
    app_config[key] = value
    save_config(app_config)
    print(f"設定を更新しました: {key} = {value}")

# --- 初期化 ---
load_config()
