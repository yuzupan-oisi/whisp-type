import pystray
from PIL import Image
import os
import sounddevice as sd
from config import app_config, update_config
from audio_processor import SAMPLE_RATE, CHANNELS

def get_mic_device_menu():
    """利用可能な入力デバイスのリストからpystrayのメニュー項目を生成する。"""
    devices = sd.query_devices()
    unique_input_devices = {}
    for i, device in enumerate(devices):
        try:
            if device['max_input_channels'] > 0:
                sd.check_input_settings(
                    device=i,
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS
                )
                key = (device['name'], device['hostapi'])
                if key not in unique_input_devices:
                    hostapi_info = sd.query_hostapis(device['hostapi'])
                    hostapi_name = hostapi_info.get('name', 'Unknown API')
                    unique_input_devices[key] = {
                        'name': device['name'],
                        'index': i,
                        'hostapi_name': hostapi_name
                    }
        except Exception:
            pass

    menu_items = []

    def create_action(value):
        return lambda: update_config("mic_device_index", value)

    def create_checked_callback(value):
        return lambda item: app_config["mic_device_index"] == value

    menu_items.append(pystray.MenuItem(
        "デフォルト",
        create_action(None),
        checked=create_checked_callback(None)
    ))

    if not unique_input_devices:
        menu_items.append(pystray.MenuItem(
            "利用可能なマイクがありません",
            None,
            enabled=False
        ))
    else:
        sorted_devices = sorted(
            unique_input_devices.values(), key=lambda d: d['name']
        )
        for device_info in sorted_devices:
            index = device_info['index']
            display_name = f"{device_info['name']} ({device_info['hostapi_name']})"
            menu_items.append(pystray.MenuItem(
                display_name,
                create_action(index),
                checked=create_checked_callback(index)
            ))
            
    return menu_items

def create_tray_icon(listener):
    icon_image = Image.open("whisp_type_icon.png")

    def on_quit():
        print("プログラムを終了します。")
        listener.stop()
        icon.stop()

    icon = pystray.Icon(
        name="WhispType",
        icon=icon_image,
        title="WhispType",
        menu=pystray.Menu(
            pystray.MenuItem("設定", pystray.Menu(
                pystray.MenuItem("言語", pystray.Menu(
                    pystray.MenuItem("日本語", lambda: update_config("language", "ja"), checked=lambda item: app_config["language"] == "ja"),
                    pystray.MenuItem("英語", lambda: update_config("language", "en"), checked=lambda item: app_config["language"] == "en")
                )),
                pystray.MenuItem("マイクデバイス", pystray.Menu(get_mic_device_menu)),
                pystray.MenuItem("フィラー語リストを開く", lambda: os.startfile(app_config["filler_words_file"])),
            )),
            pystray.MenuItem("終了", on_quit)
        )
    )
    return icon
