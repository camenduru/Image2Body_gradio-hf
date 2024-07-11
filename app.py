import gradio as gr
import firebase_admin
from firebase_admin import credentials, storage
import tempfile
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

# Firebase の初期化
cred = credentials.Certificate("firebase/image2body-demo-firebase-adminsdk-ope1k-e19e1da82c.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'image2body-demo.appspot.com'
})
def save_image_pair_to_firebase(sotai_image_data, sketch_image_data):
    # 一意の識別子を生成
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    folder_name = f"image_pairs/{timestamp}_{unique_id}"

    # Base64 データを PIL Image に変換
    sotai_image = Image.open(io.BytesIO(base64.b64decode(sotai_image_data)))
    sketch_image = Image.open(io.BytesIO(base64.b64decode(sketch_image_data)))
    
    bucket = storage.bucket()
    urls = {}

    for image_type, image in [("sotai", sotai_image), ("sketch", sketch_image)]:
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        try:
            # Firebase Storage にアップロード
            blob_path = f'{folder_name}/{image_type}.png'
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(temp_file_path)
            
            # URLを取得
            blob.make_public()
            urls[image_type] = blob.public_url
        finally:
            # 一時ファイルを削除
            os.unlink(temp_file_path)
    
    return urls

def process_image(input_image, mode, weight1, weight2):
    # 既存の画像処理ロジック
    sotai_image, sketch_image = process_image_as_base64(input_image, mode, weight1, weight2)
    
    # Firebase に画像ペアを保存し、URLを取得
    urls = save_image_pair_to_firebase(sotai_image, sketch_image)
    
    return urls['sotai'], urls['sketch']

# Gradio インターフェースの定義
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Radio(["original", "refine"], label="Mode", value="original"),
        gr.Slider(0, 2, value=0.6, step=0.05, label="Weight 1 (Sketch)"),
        gr.Slider(0, 1, value=0.05, step=0.025, label="Weight 2 (Body)")
    ],
    outputs=[
        gr.Text(label="Sotai Image URL"),
        gr.Text(label="Sketch Image URL")
    ],
    title="Image2Body API",
    description="Upload an image and select processing options to generate body and sketch images."
)

# Hugging Face Spacesでデプロイする場合
iface.queue().launch()