import gradio as gr
import spaces

with gr.Blocks() as demo:
    # title
    gr.HTML("<h1>Image2Body demo</h1>")
    # description
    gr.HTML("<p>Upload an image and select processing options to generate body and sketch images.</p>")

    demo.launch()