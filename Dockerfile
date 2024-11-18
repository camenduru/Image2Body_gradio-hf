# ベースイメージとしてPython 3.9を使用
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なPythonライブラリをインストールするための依存ファイルをコピー
COPY requirements.txt /app/requirements.txt

# 必要なPythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコンテナにコピー
COPY . /app

# ポート設定（Gradioのデフォルトポート7860）
EXPOSE 7860

# アプリケーションを起動
CMD ["python", "app.py"]
