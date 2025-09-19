# LLaVA バッチ検査環境

## 概要
本リポジトリは LLaVA 1.5 7B ビジョン・ランゲージモデルを vLLM の OpenAI 互換 API として提供し、焼却施設向けの処理不適物検査を自動化するバッチクライアントを同梱しています。ローカル GPU 上で API サーバーを立ち上げ、`client/` ディレクトリのスクリプトで大量の画像を一括判定できます。

## 必要環境
- Docker 24 以降（Compose プラグイン必須）
- 24 GB 以上の VRAM を持つ NVIDIA GPU（CUDA 12.1 ランタイムで検証済み）
- NVIDIA Container Toolkit がセットアップ済みであること
- 任意: Hugging Face で `llava-hf` モデルへアクセスできるトークン

## ディレクトリ構成
- `docker-compose.yml` : API サーバーを `http://localhost:9000/v1` で公開する Compose 設定
- `vllm/` : vLLM サーバー用 Docker ビルドコンテキスト
- `client/` : 画像バッチ検査スクリプトと設定ファイル
- `data/` : 入力 (`in/`)、JSON 出力 (`json/`)、検出結果 (`out_true/`, `out_false/`) の既定パス
- `hf_cache/` : モデルダウンロードを共有する Hugging Face キャッシュ
- `logs/`, `vllm/entrypoint.sh` など : 補助スクリプト
- `bu/` : 旧構成（レガシー用途）。**GitHub へはアップロードしないでください。必要に応じて `.gitignore` で除外してください。**

## セットアップ手順
1. リポジトリをクローンしプロジェクトルートへ移動します。
2. 必要であれば `export HF_TOKEN=...` で Hugging Face トークンをエクスポートします。
3. vLLM イメージをビルド: `docker compose build`
4. サービス起動: `docker compose up -d`
5. ヘルスチェックを確認: `docker compose logs -f vllm`
6. API 動作確認: `curl http://localhost:9000/v1/models`
7. 終了する際は `docker compose down`

### モデルキャッシュについて
- 初回起動時は `hf_cache/` にモデルがダウンロードされます。再ダウンロードを避けるためディレクトリを保持してください。
- `data/` ディレクトリはコンテナ内で `/data` として参照されます。

## バッチ検査クライアントの使い方 (`client/`)
1. 任意の Python 環境を用意します（例: `python -m venv .venv && source .venv/bin/activate`）。
2. 依存関係をインストール: `pip install -r client/requirements.txt`
3. `client/config.yaml` を編集し、API URL やモデル名、判定ルールを必要に応じて調整します。
4. 判定したい JPEG 画像を `data/in/` 以下へ配置します。
5. `python client/batch_vision.py` を実行すると、以下が生成されます。
   - 判定 JSON: `data/json/<ファイル名>.json`
   - 判定結果画像: `data/out_true/` (検知あり) と `data/out_false/` (検知なし)。検知画像には赤枠とラベルが描画されます。

## よくあるトラブルと対処
- **ダウンロード失敗**: `HF_TOKEN` がコンテナ内に渡っているか `docker compose exec vllm env | grep HF_TOKEN` で確認してください。
- **GPU メモリ不足**: `docker-compose.yml` の `--gpu-memory-utilization` や `--max-num-seqs` を調整します。
- **API エラー**: `logs/` 内のログ、または `docker compose logs vllm` で詳細を確認してください。

## 注意事項
- `bu/` ディレクトリはデバッグ目的の旧スタックです。GitHub 等の共有リポジトリには含めないでください。
- 生成結果やログには個人情報が含まれる可能性があるため、取り扱いに注意してください。

