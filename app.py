import gradio as gr
import io
from PIL import Image
import base64
from scripts.process_utils import initialize, process_image_as_base64, image_to_base64
from scripts.anime import init_model
from scripts.generate_prompt import load_wd14_tagger_model
import webbrowser

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

    return sotai_image, sketch_image, None

def mix_images(sotai_image_data, sketch_image_data, opacity1, opacity2):
    sotai_image = Image.open(io.BytesIO(base64.b64decode(sotai_image_data))).convert('RGBA')
    sketch_image = Image.open(io.BytesIO(base64.b64decode(sketch_image_data))).convert('RGBA')

    if sotai_image.size != sketch_image.size:
        sketch_image = sketch_image.resize(sotai_image.size, Image.Resampling.LANCZOS)

    mixed_image = Image.new('RGBA', sotai_image.size, (255, 255, 255, 255))

    sotai_alpha = sotai_image.getchannel('A').point(lambda x: int(x * opacity1))
    sketch_alpha = sketch_image.getchannel('A').point(lambda x: int(x * opacity2))

    mixed_image.paste(sketch_image, (0, 0), mask=sketch_alpha)
    mixed_image.paste(sotai_image, (0, 0), mask=sotai_alpha)

    return mixed_image

# X(Twitter)に投稿するリンクを生成
def generate_twitter_link(image):
    image_base64 = image_to_base64(image)
    return f"https://twitter.com/intent/tweet?text=Image2Body&url={image_base64} #Image2Body"

def post_to_twitter(image):
    link = generate_twitter_link(image)
    webbrowser.open_new_tab(link)

with gr.Blocks() as demo:
    # title
    gr.HTML("<h1>Image2Body demo</h1>")
    # description
    gr.HTML("<p>Upload an image and select processing options to generate body and sketch images.</p>")
    # interface
    submit = None
    with gr.Row():
        with gr.Column() as input_col:
            input_image = gr.Image(type="pil", label="Input Image")
            with gr.Tab("original"):
                original_mode = gr.Text("original", label="Mode", visible=False)
                original_submit = gr.Button("Submit", variant="primary")
            with gr.Tab("refine"):
                refine_input = [
                    gr.Text("refine", label="Mode", visible=False),
                    gr.Slider(0, 2, value=0.6, step=0.05, label="Weight 1 (Sketch)"),
                    gr.Slider(0, 1, value=0.05, step=0.025, label="Weight 2 (Body)")
                ]
                refine_submit = gr.Button("Submit", variant="primary")
            gr.Examples(
                examples=[f"images/sample{i}.png" for i in [1, 2, 4, 5, 6, 7, 10, 16, 18, 19]],
                inputs=[input_image]
            )
        with gr.Column() as output_col:
            sotai_image_data = gr.Text(label="Sotai Image data", visible=False)
            sketch_image_data = gr.Text(label="Sketch Image data", visible=False)
            mixed_image = gr.Image(label="Output Image", elem_id="output_image")
            opacity_slider1 = gr.Slider(0, 1, value=0.5, step=0.05, label="Opacity (Sotai)")
            opacity_slider2 = gr.Slider(0, 1, value=0.5, step=0.05, label="Opacity (Sketch)")
            # post_button = gr.Button("Post to X(Twitter)", variant="secondary")
            # post_button.click(
            #     post_to_twitter,
            #     inputs=[mixed_image],
            #     outputs=None
            # )

    original_submit.click(
        process_image,
        inputs=[input_image, original_mode],
        outputs=[sotai_image_data, sketch_image_data, mixed_image]
    )
    refine_submit.click(
        process_image,
        inputs=[input_image, refine_input[0], refine_input[1], refine_input[2]],
        outputs=[sotai_image_data, sketch_image_data, mixed_image]
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