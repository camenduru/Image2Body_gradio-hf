import gradio as gr
import io
from PIL import Image
import base64
from scripts.process_utils import initialize, process_image_as_base64, image_to_base64
from scripts.anime import init_model
from scripts.generate_prompt import load_wd14_tagger_model

# 初期化
initialize(_use_local=False, use_gpu=True, use_dotenv=True)
init_model(use_local=False)
load_wd14_tagger_model()

def process_image(input_image, mode, weight1=None, weight2=None):
    print(f"Processing image with mode={mode}, weight1={weight1}, weight2={weight2}")
    # 既存の画像処理ロジック
    if mode == "original":
        sotai_image, sketch_image = process_image_as_base64(input_image, mode, None, None)
    elif mode == "refine":
        sotai_image, sketch_image = process_image_as_base64(input_image, mode, weight1, weight2)

    # テスト用に、Base64データを返す
    sotai_image = image_to_base64(input_image)
    sketch_image = image_to_base64(input_image)

    return sotai_image, sketch_image

def mix_images(sotai_image_data, sketch_image_data, opacity1, opacity2):
    # Base64からPILイメージに変換
    sotai_image = Image.open(io.BytesIO(base64.b64decode(sotai_image_data)))
    sketch_image = Image.open(io.BytesIO(base64.b64decode(sketch_image_data)))
    # 画像を合成
    mixed_image = Image.new('RGBA', sotai_image.size, (0, 0, 0, 0))
    opacity_mask1 = Image.new('L', sotai_image.size, int(opacity1 * 255))
    opacity_mask2 = Image.new('L', sotai_image.size, int(opacity2 * 255))
    mixed_image.paste(sotai_image, (0, 0), mask=opacity_mask1)
    mixed_image.paste(sketch_image, (0, 0), mask=opacity_mask2)

    return mixed_image

with gr.Blocks() as demo:
    # title
    gr.HTML("<h1>Image2Body demo</h1>")
    # description
    gr.HTML("<p>Upload an image and select processing options to generate body and sketch images.</p>")
    # interface
    submit = None
    with gr.Row():
        with gr.Column() as input_col:
            with gr.Tab("original"):
                original_input = [
                    gr.Image(type="pil", label="Input Image"),
                    gr.Text("original", label="Mode", visible=False),
                ]
                original_submit = gr.Button("Submit", variant="primary")
            with gr.Tab("refine"):
                refine_input = [
                    gr.Image(type="pil", label="Input Image"),
                    gr.Text("refine", label="Mode", visible=False),
                    gr.Slider(0, 2, value=0.6, step=0.05, label="Weight 1 (Sketch)"),
                    gr.Slider(0, 1, value=0.05, step=0.025, label="Weight 2 (Body)")
                ]
                refine_submit = gr.Button("Submit", variant="primary")
        with gr.Column() as output_col:
            sotai_image_data = gr.Text(label="Sotai Image data", visible=False)
            sketch_image_data = gr.Text(label="Sketch Image data", visible=False)
            mixed_image = gr.Image(label="Output Image", elem_id="output_image")
            opacity_slider1 = gr.Slider(0, 1, value=0.5, step=0.05, label="Opacity (Sotai)")
            opacity_slider2 = gr.Slider(0, 1, value=0.5, step=0.05, label="Opacity (Sketch)")

    original_submit.click(
        process_image,
        inputs=original_input,
        outputs=[sotai_image_data, sketch_image_data]
    )
    refine_submit.click(
        process_image,
        inputs=refine_input,
        outputs=[sotai_image_data, sketch_image_data]
    )
    sotai_image_data.change(
        mix_images,
        inputs=[sotai_image_data, sketch_image_data, opacity_slider1, opacity_slider2],
        outputs=mixed_image
    )
    opacity_slider1.change(
        mix_images,
        inputs=[sotai_image_data, sketch_image_data, opacity_slider1, opacity_slider2],
        outputs=mixed_image
    )
    opacity_slider2.change(
        mix_images,
        inputs=[sotai_image_data, sketch_image_data, opacity_slider1, opacity_slider2],
        outputs=mixed_image
    )

    demo.launch()


# # Gradio インターフェースの定義
# iface = gr.Interface(
#     fn=process_image,
#     inputs=[
#         gr.Image(type="pil", label="Input Image"),
#         gr.Radio(["original", "refine"], label="Mode", value="original"),
#         gr.Slider(0, 2, value=0.6, step=0.05, label="Weight 1 (Sketch)"),
#         gr.Slider(0, 1, value=0.05, step=0.025, label="Weight 2 (Body)")
#     ],
#     outputs=[
#         gr.Text(label="Sotai Image URL"),
#         gr.Text(label="Sketch Image URL")
#     ],
#     title="Image2Body API",
#     description="Upload an image and select processing options to generate body and sketch images."
# )

# # Hugging Face Spacesでデプロイする場合
# iface.queue().launch()