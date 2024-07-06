# CUDA 12.1ベースのUbuntuイメージを使用
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    wget \
    && rm -rf /var/lib/apt/lists/*
# pipのインストール
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py
# デフォルトのpythonとpipコマンドをpython3.10とpip3.10にリンク
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

WORKDIR /app

RUN useradd -m -u 1000 user
USER user

# 依存関係をインストール
COPY requirements.txt /app/
RUN apt -y update && apt -y upgrade
RUN apt -y install libopencv-dev
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-dependencies transformers

COPY . /app

EXPOSE 80

CMD ["python", "app.py", "--use_gpu"]
