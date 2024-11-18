# CUDA 12.1ベースのUbuntuイメージを使用
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
FROM python:3.10-slim

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

WORKDIR /app

# 依存関係をインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-dependencies transformers

# アプリケーションのコピー
COPY . /app

# 非rootユーザーの作成と切り替え
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 起動コマンドを変更
CMD ["python", "app.py", "--use_gpu"]