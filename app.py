import gradio as gr
import os
import io
from PIL import Image
import base64
from scripts.process_utils import initialize, process_image_as_base64
from scripts.anime import init_model
from scripts.generate_prompt import load_wd14_tagger_model

# 初期化
initialize(_use_local=False, use_gpu=True)
init_model(use_local=False)
load_wd14_tagger_model()

def process_image(input_image, mode, weight1, weight2):
    # 画像処理ロジック
    sotai_image, sketch_image = process_image_as_base64(input_image, mode, weight1, weight2)
    
    # Base64文字列をPIL Imageに変換
    sotai_pil = Image.open(io.BytesIO(base64.b64decode(sotai_image)))
    sketch_pil = Image.open(io.BytesIO(base64.b64decode(sketch_image)))
    
    return sotai_pil, sketch_pil

def gradio_process_image(input_image, mode, weight1, weight2):
    sotai_image, sketch_image = process_image(input_image, mode, weight1, weight2)
    return sotai_image, sketch_image

# サンプル画像のパスリスト
sample_images = [
  '/images/sample1.png',
  '/images/sample2.png',
  # '/images/sample3.png',
  '/images/sample4.png',
  '/images/sample5.png',
  '/images/sample6.png',
  '/images/sample7.png',
  '/images/sample8.png',
  # '/images/sample9.png',
  '/images/sample10.png',
  '/images/sample11.png',
  # '/images/sample12.png',
  # '/images/sample13.png',
  # '/images/sample14.png',
  '/images/sample15.png',
  '/images/sample16.png',
  # '/images/sample17.png',
  '/images/sample18.png',
  '/images/sample19.png',
  '/images/sample20.png',
  '/images/sample21.png',
]

# Gradio インターフェースの定義
with gr.Blocks() as demo:
    gr.Markdown("# Image2Body Test")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            mode = gr.Radio(["original", "refine"], label="Mode", value="original")
            with gr.Row():
                weight1 = gr.Slider(0, 2, value=0.6, step=0.05, label="Weight 1 (Sketch)")
                weight2 = gr.Slider(0, 1, value=0.05, step=0.025, label="Weight 2 (Body)")
            process_btn = gr.Button("Process")
        
        with gr.Column():
            sotai_output = gr.Image(type="pil", label="Sotai (Body) Image")
            sketch_output = gr.Image(type="pil", label="Sketch Image")
    
    gr.Examples(
        examples=sample_images,
        inputs=input_image,
        outputs=[sotai_output, sketch_output],
        fn=gradio_process_image,
        cache_examples=True,
    )
    
    process_btn.click(
        fn=gradio_process_image,
        inputs=[input_image, mode, weight1, weight2],
        outputs=[sotai_output, sketch_output]
    )

# Spacesへのデプロイ設定
demo.launch()