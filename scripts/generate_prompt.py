import argparse
import csv
import os
import json

from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.layers import TFSMLayer
from huggingface_hub import hf_hub_download
from pathlib import Path

# 画像サイズの設定
IMAGE_SIZE = 448

# デフォルトのタグ付けリポジトリとファイル構成
DEFAULT_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
MODEL_FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
VAR_DIR = "variables"
VAR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = MODEL_FILES[-1]

def preprocess_image(image):
    """画像を前処理して正方形に変換"""
    img = np.array(image)[:, :, ::-1]  # RGB->BGR

    size = max(img.shape[:2])
    pad_x, pad_y = size - img.shape[1], size - img.shape[0]
    img = np.pad(img, ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    return img.astype(np.float32)

def download_model_files(repo_id, model_dir, sub_dir, files, sub_files):
    """モデルファイルをHugging Face Hubからダウンロード"""
    for file in files:
        hf_hub_download(repo_id, file, cache_dir=model_dir, force_download=True, force_filename=file)
    for file in sub_files:
        hf_hub_download(repo_id, file, subfolder=sub_dir, cache_dir=os.path.join(model_dir, sub_dir), force_download=True, force_filename=file)

def load_wd14_tagger_model():
    """WD14タグ付けモデルをロード"""
    model_dir = "wd14_tagger_model"
    if not os.path.exists(model_dir):
        download_model_files(DEFAULT_REPO, model_dir, VAR_DIR, MODEL_FILES, VAR_FILES)
    else:
        print("Using existing model")
    return TFSMLayer(model_dir, call_endpoint='serving_default')

def read_tags_from_csv(csv_path):
    """CSVファイルからタグを読み取る"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        tags = [row for row in reader]
    header = tags[0]
    rows = tags[1:]
    assert header[:3] == ["tag_id", "name", "category"], f"Unexpected CSV format: {header}"
    return rows

def generate_tags(images, model_dir, model):
    """画像にタグを生成"""
    rows = read_tags_from_csv(os.path.join(model_dir, CSV_FILE))
    general_tags = [row[1] for row in rows if row[2] == "0"]
    character_tags = [row[1] for row in rows if row[2] == "4"]
    
    tag_freq = {}
    undesired_tags = {'one-piece_swimsuit', 'swimsuit', 'leotard', 'saitama_(one-punch_man)', '1boy'}

    probs = model(images, training=False)['predictions_sigmoid'].numpy()
    tag_text_list = []

    for prob in probs:
        tags_combined = []
        for i, p in enumerate(prob[4:]):
            tag_list = general_tags if i < len(general_tags) else character_tags
            tag = tag_list[i - len(general_tags)] if i >= len(general_tags) else tag_list[i]
            if p >= 0.35 and tag not in undesired_tags:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
                tags_combined.append(tag)

        tag_text_list.append(", ".join(tags_combined))
    return tag_text_list
