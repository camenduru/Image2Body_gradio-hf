import gradio as gr
import os
import io
from PIL import Image
import base64
from scripts.process_utils import initialize, process_image_as_base64
from scripts.anime import init_model
from scripts.generate_prompt import load_wd14_tagger_model

# 初期化
initialize(_use_local=False, use_gpu=True, use_dotenv=False)
init_model(use_local=False)
load_wd14_tagger_model()

def process_image(input_image, mode, weight1, weight2):
    # 画像処理ロジック
    sotai_image, sketch_image = process_image_as_base64(input_image, mode, weight1, weight2)
    return sotai_image, sketch_image

def gradio_process_image(input_image, mode, weight1, weight2):
    # Gradio用の関数：PILイメージを受け取り、Base64文字列を返す
    input_image_bytes = io.BytesIO()
    input_image.save(input_image_bytes, format='PNG')
    input_image_base64 = base64.b64encode(input_image_bytes.getvalue()).decode('utf-8')
    
    sotai_base64, sketch_base64 = process_image(input_image_base64, mode, weight1, weight2)
    return sotai_base64, sketch_base64

# Gradio インターフェースの定義
iface = gr.Interface(
    fn=gradio_process_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Radio(["original", "refine"], label="Mode", value="original"),
        gr.Slider(0, 2, value=0.6, step=0.05, label="Weight 1 (Sketch)"),
        gr.Slider(0, 1, value=0.05, step=0.025, label="Weight 2 (Body)")
    ],
    outputs=[
        gr.Image(type="pil", label="Sotai (Body) Image"),
        gr.Image(type="pil", label="Sketch Image")
    ],
    title="Image2Body API",
    description="Upload an image and select processing options to generate body and sketch images."
)

# # APIとして公開
# app = gr.mount_gradio_app(app, iface, path="/")

# Hugging Face Spacesでデプロイする場合
iface.queue().launch()