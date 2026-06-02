# ベースとなる軽量なPython環境を指定
FROM python:3.10-slim

# 音声処理(librosa)に必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spacesのセキュリティ要件（ユーザー権限の設定）
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# ローカルのファイルをすべてコンテナの中にコピー
COPY --chown=user . $HOME/app

# pipのアップグレードとPyTorch（軽量なCPU版）のインストール
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# requirements.txtのライブラリを一括インストール
RUN pip install --no-cache-dir -r requirements.txt

# 起動スクリプトに実行権限を付与して実行
RUN chmod +x run.sh
CMD ["./run.sh"]