import io
import os
import base64
from PIL import Image
import cv2
import numpy as np
from scripts.generate_prompt import load_wd14_tagger_model, generate_tags, preprocess_image as wd14_preprocess_image
from scripts.lineart_util import scribble_xdog, get_sketch, canny
from scripts.anime import init_model
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
import gc
from peft import PeftModel
from dotenv import load_dotenv
from scripts.hf_utils import download_file

import spaces

# グローバル変数
use_local = False
model = None
device = None
torch_dtype = None # torch.float16 if device == "cuda" else torch.float32
sotai_gen_pipe = None
refine_gen_pipe = None

def get_file_path(filename, subfolder):
    if use_local:
        return subfolder + "/" + filename
    else:
        return download_file(filename, subfolder)

def ensure_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def initialize(_use_local=False, use_gpu=False, use_dotenv=False):
    if use_dotenv:
        load_dotenv()
    global model, sotai_gen_pipe, refine_gen_pipe, use_local, device, torch_dtype
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    use_local = _use_local

    print(f"\nDevice: {device}, Local model: {_use_local}\n")

    init_model(use_local)
    model = load_wd14_tagger_model()
    sotai_gen_pipe = initialize_sotai_model()
    refine_gen_pipe = initialize_refine_model()

def load_lora(pipeline, lora_path, adapter_name, alpha=0.75):
    pipeline.load_lora_weights(lora_path, adapter_name)
    pipeline.fuse_lora(lora_scale=alpha, adapter_names=[adapter_name])
    pipeline.set_lora_device(adapter_names=[adapter_name], device=device)   

def initialize_sotai_model():
    global device, torch_dtype

    sotai_sd_model_path = get_file_path(os.environ["sotai_sd_model_name"], subfolder=os.environ["sd_models_dir"])
    controlnet_path1 =  get_file_path(os.environ["controlnet_name1"], subfolder=os.environ["controlnet_dir2"])
    # controlnet_path1 =  get_file_path(os.environ["controlnet_name2"], subfolder=os.environ["controlnet_dir1"])
    controlnet_path2 =  get_file_path(os.environ["controlnet_name2"], subfolder=os.environ["controlnet_dir1"])

    # Load the Stable Diffusion model
    sd_pipe = StableDiffusionPipeline.from_single_file(
        sotai_sd_model_path,
        torch_dtype=torch_dtype,
        use_safetensors=True
    ).to(device)
    
    # Load the ControlNet model
    controlnet1 = ControlNetModel.from_single_file(
        controlnet_path1,
        torch_dtype=torch_dtype
    ).to(device)
    
    # Load the ControlNet model
    controlnet2 = ControlNetModel.from_single_file(
        controlnet_path2,
        torch_dtype=torch_dtype
    ).to(device)

    # Create the ControlNet pipeline
    sotai_gen_pipe = StableDiffusionControlNetPipeline(
        vae=sd_pipe.vae,
        text_encoder=sd_pipe.text_encoder,
        tokenizer=sd_pipe.tokenizer,
        unet=sd_pipe.unet,
        scheduler=sd_pipe.scheduler,
        safety_checker=sd_pipe.safety_checker,
        feature_extractor=sd_pipe.feature_extractor,
        controlnet=[controlnet1, controlnet2]
    ).to(device)

    # LoRAの適用
    lora_names = [
        (os.environ["lora_name1"], 1.0),
        # (os.environ["lora_name2"], 0.3),
    ]
    
    for lora_name, alpha in lora_names:
        lora_path = get_file_path(lora_name, subfolder=os.environ["lora_dir"])
        load_lora(sotai_gen_pipe, lora_path, adapter_name=lora_name.split(".")[0], alpha=alpha)

    # スケジューラーの設定
    sotai_gen_pipe.scheduler = UniPCMultistepScheduler.from_config(sotai_gen_pipe.scheduler.config)

    return sotai_gen_pipe

def initialize_refine_model():
    global device, torch_dtype

    refine_sd_model_path = get_file_path(os.environ["refine_sd_model_name"], subfolder=os.environ["sd_models_dir"])
    controlnet_path3 = get_file_path(os.environ["controlnet_name3"], subfolder=os.environ["controlnet_dir1"])
    controlnet_path4 = get_file_path(os.environ["controlnet_name4"], subfolder=os.environ["controlnet_dir1"])
    vae_path = get_file_path(os.environ["vae_name"], subfolder=os.environ["vae_dir"])

    # Load the Stable Diffusion model
    sd_pipe = StableDiffusionPipeline.from_single_file(
        refine_sd_model_path,
        torch_dtype=torch_dtype,
        variant="fp16", 
        use_safetensors=True
    ).to(device)
    
    # controlnet_path = "models/cn/control_v11p_sd15_canny.pth"
    controlnet1 = ControlNetModel.from_single_file(
        controlnet_path3,
        torch_dtype=torch_dtype
    ).to(device)
    
    # Load the ControlNet model
    controlnet2 = ControlNetModel.from_single_file(
        controlnet_path4,
        torch_dtype=torch_dtype
    ).to(device)

    # Create the ControlNet pipeline
    refine_gen_pipe = StableDiffusionControlNetPipeline(
        vae=AutoencoderKL.from_single_file(vae_path, torch_dtype=torch_dtype).to(device),
        text_encoder=sd_pipe.text_encoder,
        tokenizer=sd_pipe.tokenizer,
        unet=sd_pipe.unet,
        scheduler=sd_pipe.scheduler,
        safety_checker=sd_pipe.safety_checker,
        feature_extractor=sd_pipe.feature_extractor,
        controlnet=[controlnet1, controlnet2],  # 複数のControlNetを指定
    ).to(device)

    # スケジューラーの設定
    refine_gen_pipe.scheduler = UniPCMultistepScheduler.from_config(refine_gen_pipe.scheduler.config)

    return refine_gen_pipe

def get_wd_tags(images: list) -> list:
    global model
    if model is None:
        raise ValueError("Model is not initialized")
        # initialize()
    preprocessed_images = [wd14_preprocess_image(img) for img in images]
    preprocessed_images = np.array(preprocessed_images)
    return generate_tags(preprocessed_images, os.environ["wd_model_name"], model)

def preprocess_image_for_generation(image):
    if isinstance(image, str):  # base64文字列の場合
        image = Image.open(io.BytesIO(base64.b64decode(image)))
    elif isinstance(image, np.ndarray):  # numpy配列の場合
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Unsupported image type")
    
    # 画像サイズの計算
    input_width, input_height = image.size
    max_size = 736
    output_width = max_size if input_height < input_width else int(input_width / input_height * max_size)
    output_height = max_size if input_height > input_width else int(input_height / input_width * max_size)
    
    image = image.resize((output_width, output_height))
    return image, output_width, output_height

def binarize_image(image: Image.Image) -> np.ndarray:
    image = np.array(image.convert('L'))
    # 色反転
    image = 255 - image
    
    # ヒストグラム平坦化
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # ガウシアンブラー適用
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 適応的二値化
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -8)

    return binary_image

def create_rgba_image(binary_image: np.ndarray, color: list) -> Image.Image:
    rgba_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, 0] = color[0]
    rgba_image[:, :, 1] = color[1]
    rgba_image[:, :, 2] = color[2]
    rgba_image[:, :, 3] = binary_image
    return Image.fromarray(rgba_image, 'RGBA')

@spaces.GPU
def generate_sotai_image(input_image: Image.Image, output_width: int, output_height: int) -> Image.Image:
    input_image = ensure_rgb(input_image)
    global sotai_gen_pipe
    if sotai_gen_pipe is None:
        raise ValueError("Model is not initialized")
        # initialize()

    prompt = "anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)"
    try:
        # 入力画像のリサイズ
        if input_image.size[0] > input_image.size[1]:
            input_image = input_image.resize((512, int(512 * input_image.size[1] / input_image.size[0])))
        else:
            input_image = input_image.resize((int(512 * input_image.size[0] / input_image.size[1]), 512))

        # EasyNegativeV2の内容
        easy_negative_v2 = "(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name, (bad_prompt_version2:0.8)"

        output = sotai_gen_pipe(
            prompt,
            image=[input_image, input_image],
            negative_prompt=f"(wings:1.6), (clothes, garment, lighting, gray, missing limb, extra line, extra limb, extra arm, extra legs, hair, bangs, fringe, forelock, front hair, fill:1.4), (ink pool:1.6)",
            # negative_prompt=f"{easy_negative_v2}, (wings:1.6), (clothes, garment, lighting, gray, missing limb, extra line, extra limb, extra arm, extra legs, hair, bangs, fringe, forelock, front hair, fill:1.4), (ink pool:1.6)",
            num_inference_steps=20,
            guidance_scale=8,
            width=output_width,
            height=output_height,
            denoising_strength=0.13,
            num_images_per_prompt=1,  # Equivalent to batch_size
            guess_mode=[True, True],  # Equivalent to pixel_perfect
            controlnet_conditioning_scale=[1.4, 1.3],  # 各ControlNetの重み
            guidance_start=[0.0, 0.0],
            guidance_end=[1.0, 1.0],
        )
        generated_image = output.images[0]
        
        return generated_image

    finally:
        # メモリ解放
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

@spaces.GPU
def generate_refined_image(prompt: str, original_image: Image.Image, output_width: int, output_height: int, weight1: float, weight2: float) -> Image.Image:
    original_image = ensure_rgb(original_image)
    global refine_gen_pipe
    if refine_gen_pipe is None:
        raise ValueError("Model is not initialized")
        # initialize()

    try:
        original_image_np = np.array(original_image)
        # scribble_xdog
        scribble_image, _ = scribble_xdog(original_image_np, 2048, 20)

        original_image = original_image.resize((output_width, output_height))
        output = refine_gen_pipe(
            prompt,
            image=[scribble_image, original_image],  # 2つのControlNetに対応する入力画像
            negative_prompt="extra limb, monochrome, black and white",
            num_inference_steps=20,
            width=output_width,
            height=output_height,
            controlnet_conditioning_scale=[weight1, weight2],  # 各ControlNetの重み
            control_guidance_start=[0.0, 0.0],
            control_guidance_end=[1.0, 1.0],
            guess_mode=[False, False],  # pixel_perfect
        )
        generated_image = output.images[0]
        
        return generated_image

    finally:
        # メモリ解放
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def process_image(input_image, mode: str, weight1: float = 0.4, weight2: float = 0.3):
    input_image = ensure_rgb(input_image)
    # サイズを取得
    input_width, input_height = input_image.size
    max_size = 736
    output_width = max_size if input_height < input_width else int(input_width / input_height * max_size)
    output_height = max_size if input_height > input_width else int(input_height / input_width * max_size)

    if mode == "refine":
        # WD-14 taggerを使用してプロンプトを生成
        image_np = np.array(ensure_rgb(input_image))
        prompt = get_wd_tags([image_np])[0]
        prompt = f"{prompt}"

        refined_image = generate_refined_image(prompt, input_image, output_width, output_height, weight1, weight2)
        refined_image = refined_image.convert('RGB')

        # スケッチ画像を生成
        refined_image_np = np.array(refined_image)
        sketch_image = get_sketch(refined_image_np, "both", 2048, 10)
        sketch_image = sketch_image.resize((output_width, output_height))  # 画像サイズを合わせる
        # スケッチ画像の二値化
        sketch_binary = binarize_image(sketch_image)
        # RGBAに変換（透明なベース画像を作成）して、青い線を設定
        sketch_image = create_rgba_image(sketch_binary, [0, 0, 255])

        # 素体画像の生成
        sotai_image = generate_sotai_image(refined_image, output_width, output_height)

    elif mode == "original":
        sotai_image = generate_sotai_image(input_image, output_width, output_height)
        
        # スケッチ画像の生成
        input_image_np = np.array(input_image)
        sketch_image = get_sketch(input_image_np, "both", 2048, 16)

    elif mode == "sketch":
        # スケッチ画像の生成
        input_image_np = np.array(input_image)
        sketch_image = get_sketch(input_image_np, "both", 2048, 16)
        
        # 素体画像の生成
        sotai_image = generate_sotai_image(sketch_image, output_width, output_height)

    else:
        raise ValueError("Invalid mode")

    # 素体画像の二値化
    sotai_binary = binarize_image(sotai_image)
    # RGBAに変換（透明なベース画像を作成）して、赤い線を設定
    sotai_image = create_rgba_image(sotai_binary, [255, 0, 0])

    return sotai_image, sketch_image

def image_to_base64(img_array):
    buffered = io.BytesIO()
    img_array.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_image_as_base64(input_image, mode: str, weight1: float = 0.4, weight2: float = 0.3):
    sotai_image, sketch_image = process_image(input_image, mode, weight1, weight2)
    return image_to_base64(sotai_image), image_to_base64(sketch_image)
