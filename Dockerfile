# CUDA 12.1ベースのUbuntuイメージを使用
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
FROM ubuntu:20.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# システムの更新とPython 3.10のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    libopencv-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    redis-server \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pipのインストールとPythonのセットアップ
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

WORKDIR /app

# 依存関係をインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-dependencies transformers

# アプリケーションのコピー
COPY . /app

# Redis の起動スクリプトを作成
RUN echo '#!/bin/bash\nredis-server --daemonize yes\npython app.py' > /app/start.sh \
    && chmod +x /app/start.sh

# 非rootユーザーの作成と切り替え
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 80

# 起動コマンドを変更
CMD ["/app/start.sh"]