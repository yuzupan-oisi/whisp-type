# WhispType
WhispTypeは、[faster-whisper](https://github.com/guillaumekln/faster-whisper)を利用して、リアルタイムで高精度な音声入力を行うためのWindows向け常駐型アプリケーションです。マイクからの音声を自動で文字起こしし、「えー」や「あのー」といったフィラー語を除去した上で、アクティブなウィンドウにテキストを挿入します。

このアプリケーションおよびREADMEは、**Google Gemini CLI** の支援を受けて作成されました。

また、このツールを作成したときのお話をnoteで公開していますので、ぜひご覧ください。
https://note.com/yuzupan/n/n05f9928e8f4d

## 主な機能

-   **リアルタイム文字起こし:** マイクからの音声をリアルタイムでテキストに変換します。
-   **フィラー語除去:** 設定ファイルに基づいて不要なフィラー語（「えー」「あのー」など）を自動で除去し、クリーンなテキストを生成します。
-   **ホットキーによる操作:** `Ctrl+Alt+Space` のホットキーで、いつでも音声認識の開始・停止が可能です。
-   **テキストの自動挿入:** 認識されたテキストは、現在カーソルがある位置に自動で挿入されます。
-   **システムトレイ常駐:** アプリケーションはシステムトレイに常駐し、邪魔になりません。右クリックメニューから設定変更や終了が可能です。
-   **GPU対応:** `config.json`で設定を変更することで、GPU（CUDA）を使用した高速な文字起こしが可能です。
-   **動的プロンプト:** 直前の文字起こし結果を次の認識のプロンプトとして利用し、文脈に応じた認識精度を向上させます。

## 動作環境

-   Windows OS
-   NVIDIA製GPU（CUDA対応、GPU利用時）
    -   CUDA 12.9で確認済み

## インストールと実行方法

### 1. リポジトリのクローン

```bash
git clone https://github.com/yuzupan-oisi/whisp-type.git
cd whisp-type
```

### 2. 依存ライブラリのインストール

仮想環境を作成し、必要なライブラリをインストールします。
※GPUを使用する場合、使用するCUDAのバージョンに対応したPyTorchをインストールしてください。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 実行

以下を実行すると、アプリケーションがシステムトレイに常駐して起動します。
初回文字起こし時は


```bash
python main.py
```

### 4. 終了

システムトレイを右クリックして終了を押すことで終了できます。
（現在、終了が不安定なことを確認）

## 使用方法

1.  `run_main.bat` を実行してアプリケーションを起動します。
2.  文字入力したいアプリケーション（メモ帳、ブラウザなど）を開き、カーソルを合わせます。
3.  `Ctrl+Alt+Space` を押すと、録音が開始されます。（トレイアイコンが変化します）
4.  マイクに向かって話します。無音状態が一定時間続くと、自動で文字起こしが実行されます。
5.  フィラー語が除去されたテキストが、カーソル位置に自動で挿入されます。
6.  再度 `Ctrl+Alt+Space` を押すと、録音が停止します。

## 設定

設定は `config.json` ファイルを直接編集することで変更できます。以下は各設定項目の説明です。

| キー | 型 | 説明 |
| --- | --- | --- |
| `language` | `string` | 文字起こしする言語のコードです（例: "ja", "en"）。Whisperが対応する言語を指定します。 |
| `mic_device_index` | `integer` or `null` | 使用するマイクのデバイスID。`null`に設定すると、OSのデフォルトマイクが自動的に選択されます。 |
| `filler_words_file` | `string` | 除去したいフィラー語を一行ずつ記述したテキストファイルへのパスです。 |
| `use_gpu` | `boolean` | `true`に設定すると、NVIDIA製GPU（CUDA）を使用して高速な文字起こしを行います。`false`の場合はCPUを使用します。 |
| `compute_type` | `string` | 計算に使用するデータ型（例: "int8", "float16", "float32"）。GPUの性能やVRAM容量に応じて設定します。"int8"は高速ですが、精度が若干低下する可能性があります。 |
| `long_silence_duration_s` | `float` | 長い無音と判断する秒数。この秒数以上無音が続くと、文脈（プロンプト）がリセットされます。 |
| `default_initial_prompt` | `string` | アプリケーション起動後、最初の文字起こしで使用される初期プロンプトです。句読点のスタイルや専門用語などを指定することで、認識精度を向上させることができます。 |
| `clear_clipboard_after_insert` | `boolean` | `true`に設定すると、テキスト挿入後にクリップボードの内容を空にします。セキュリティを考慮する場合に有効です。 |
| `silence_duration_s` | `float` | 発話の区切りと判断する無音の秒数。この秒数だけ無音が続くと、そこまでの音声をまとめて文字起こし処理に送ります。 |

## 注意事項

-   このソフトウェアは **Google Gemini CLI** の支援を受けて開発されたものです。
-   使用は自己責任でお願いします。本ソフトウェアの使用によって生じたいかなる損害についても、開発者は責任を負いません。
