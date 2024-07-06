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

# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image


def load_wd14_tagger_model():
    model_dir = "wd14_tagger_model"
    repo_id = DEFAULT_WD14_TAGGER_REPO

    if not os.path.exists(model_dir):
        print(f"downloading wd14 tagger model from hf_hub. id: {repo_id}")
        for file in FILES:
            hf_hub_download(repo_id, file, cache_dir=model_dir, force_download=True, force_filename=file)
        for file in SUB_DIR_FILES:
            hf_hub_download(
                repo_id,
                file,
                subfolder=SUB_DIR,
                cache_dir=os.path.join(model_dir, SUB_DIR),
                force_download=True,
                force_filename=file,
            )
    else:
        print("using existing wd14 tagger model")

    # モデルを読み込む
    model = TFSMLayer(model_dir, call_endpoint='serving_default')
    return model


def generate_tags(images, model_dir, model):
    with open(os.path.join(model_dir, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        header = l[0]  # tag_id,name,category,count
        rows = l[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    general_tags = [row[1] for row in rows[1:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[1:] if row[2] == "4"]

    tag_freq = {}
    undesired_tags = ['one-piece_swimsuit',
                      'swimsuit',
                      'leotard',
                      'saitama_(one-punch_man)',
                      '1boy',
    ]

    probs = model(images, training=False)
    probs = probs['predictions_sigmoid'].numpy()

    tag_text_list = []
    for prob in probs:
        combined_tags = []
        general_tag_text = ""
        character_tag_text = ""
        thresh = 0.35
        for i, p in enumerate(prob[4:]):
            if i < len(general_tags) and p >= thresh:
                tag_name = general_tags[i]
                if tag_name not in undesired_tags:
                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                    general_tag_text += ", " + tag_name
                    combined_tags.append(tag_name)
            elif i >= len(general_tags) and p >= thresh:
                tag_name = character_tags[i - len(general_tags)]
                if tag_name not in undesired_tags:
                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                    character_tag_text += ", " + tag_name
                    combined_tags.append(tag_name)

        if len(general_tag_text) > 0:
            general_tag_text = general_tag_text[2:]
        if len(character_tag_text) > 0:
            character_tag_text = character_tag_text[2:]

        tag_text = ", ".join(combined_tags)
        tag_text_list.append(tag_text)
    return tag_text_list
        

def generate_prompt_json(target_folder, prompt_file, model_dir, model):
    image_files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    image_count = len(image_files)

    prompt_list = []

    for i, filename in enumerate(image_files, 1):
        source_path = "source/" + filename
        target_path = os.path.join(target_folder, filename)  # Use absolute path
        target_path2 = "target/" + filename

        prompt = generate_tags(target_path, model_dir, model)

        for j in range(4):
            prompt_data = {
                "source": f"{source_path.split('.')[0]}_{j}.jpg",
                "target": f"{target_path2.split('.')[0]}_{j}.jpg",
                "prompt": prompt
            }

            prompt_list.append(prompt_data)

        print(f"Processed Images: {i}/{image_count}", end="\r", flush=True)

    with open(prompt_file, "w") as file:
        for prompt_data in prompt_list:
            json.dump(prompt_data, file)
            file.write("\n")

    print(f"Processing completed. Total Images: {image_count}")


if __name__ == '__main__':
    model_dir = "wd14_tagger_model"
    model = load_wd14_tagger_model()
    prompt = generate_tags(target_path, model_dir, model)