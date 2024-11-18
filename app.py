import gradio as gr
import spaces

def process_image(input_image, mode, weight1=None, weight2=None):
    print(f"Processing image with mode={mode}, weight1={weight1}, weight2={weight2}")
    # 既存の画像処理ロジック
    # if mode == "original":
    #     sotai_image, sketch_image = process_image_as_base64(input_image, mode, None, None)
    # elif mode == "refine":
    #     sotai_image, sketch_image = process_image_as_base64(input_image, mode, weight1, weight2)

    return input_image

with gr.Blocks() as demo:
    # title
    gr.HTML("<h1>Image2Body demo</h1>")
    # description
    gr.HTML("<p>Upload an image and select processing options to generate body and sketch images.</p>")
    # interface

    demo.launch()