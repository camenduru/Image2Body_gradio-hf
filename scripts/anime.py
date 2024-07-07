"""Test script for anime-to-sketch translation
Example:
    python3 test.py --dataroot /your_path/dir --load_size 512
    python3 test.py --dataroot /your_path/img.jpg --load_size 512
"""

import os
import torch
from scripts.data import get_image_list, get_transform, tensor_to_img, save_image
from scripts.model import create_model
import argparse
from tqdm.auto import tqdm
from kornia.enhance import equalize_clahe
from PIL import Image
import numpy as np

model = None

def init_model(use_local=False):
    global model
    model_opt = "default"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_opt, use_local).to(device)
    model.eval()

# numpy配列の画像を受け取り、線画を生成してnumpy配列で返す
def generate_sketch(image, clahe_clip=-1, load_size=512):
    """
    Generate sketch image from input image
    Args:
        image (np.ndarray): input image
        clahe_clip (float): clip threshold for CLAHE
        load_size (int): image size to load
    Returns:
        np.ndarray: output image
    """
    # create model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_opt = "default"
    # model = create_model(model_opt).to(device)
    # model.eval()
    
    aus_resize = None
    if load_size > 0:
        aus_resize = (image.shape[0], image.shape[1])
    transform = get_transform(load_size=load_size)
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    # [0,255] to [-1,1]
    image = transform(image)
    if image.max() > 1:
        image = (image-image.min())/(image.max()-image.min())*2-1

    img, aus_resize = image.unsqueeze(0), aus_resize
    if clahe_clip > 0:
        img = (img + 1) / 2 # [-1,1] to [0,1]
        img = equalize_clahe(img, clip_limit=clahe_clip)
        img = (img - .5) / .5 # [0,1] to [-1,1]

    aus_tensor = model(img.to(device))
    
    # resize to original size
    if aus_resize is not None:
        aus_tensor = torch.nn.functional.interpolate(aus_tensor, aus_resize, mode='bilinear', align_corners=False)

    aus_img = tensor_to_img(aus_tensor)
    return aus_img


if __name__ == '__main__':
    os.chdir(os.path.dirname("Anime2Sketch/"))
    parser = argparse.ArgumentParser(description='Anime-to-sketch test options.')
    parser.add_argument('--dataroot','-i', default='test_samples/', type=str)
    parser.add_argument('--load_size','-s', default=512, type=int)
    parser.add_argument('--output_dir','-o', default='results/', type=str)
    parser.add_argument('--gpu_ids', '-g', default=[], help="gpu ids: e.g. 0 0,1,2 0,2.")
    parser.add_argument('--model', default="default", type=str, help="variant of model to use. you can choose from ['default','improved']")
    parser.add_argument('--clahe_clip', default=-1, type=float, help="clip threshold for CLAHE set to -1 to disable")
    opt = parser.parse_args()

    # # generate sketchで線画生成
    # for test_path in tqdm(get_image_list(opt.dataroot)):
    #     basename = os.path.basename(test_path)
    #     aus_path = os.path.join(opt.output_dir, basename)
    #     # numpy配列で画像を読み込む
    #     img = Image.open(test_path)
    #     img = np.array(img)
    #     aus_img = generate_sketch(img, opt.clahe_clip)
    #     # 画像を保存
    #     save_image(aus_img, aus_path, (512, 512))

    
    # create model
    gpu_list = ','.join(str(x) for x in opt.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(opt.model, use_local=True).to(device)      # create a model given opt.model and other options
    model.eval()
    
    for test_path in tqdm(get_image_list(opt.dataroot)):
        basename = os.path.basename(test_path)
        aus_path = os.path.join(opt.output_dir, basename)

        img = Image.open(test_path).convert('RGB')
        img = np.array(img)

        load_size = 512
        aus_resize = None
        if load_size > 0:
            aus_resize = (img.shape[1], img.shape[0])
        transform = get_transform(load_size=load_size)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # [0,255] to [-1,1]
        image = transform(img)
        if image.max() > 1:
            image = (image-image.min())/(image.max()-image.min())*2-1
            print(image.min(), image.max())

        img, aus_resize = image.unsqueeze(0), aus_resize
        if opt.clahe_clip > 0:
            img = (img + 1) / 2 # [-1,1] to [0,1]
            img = equalize_clahe(img, clip_limit=opt.clahe_clip)
            img = (img - .5) / .5 # [0,1] to [-1,1]

        aus_tensor = model(img.to(device))
        aus_img = tensor_to_img(aus_tensor)
        save_image(aus_img, aus_path, aus_resize)
"""
    # create model
    gpu_list = ','.join(str(x) for x in opt.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    device = torch.device('cuda' if len(opt.gpu_ids)>0 else 'cpu')
    model = create_model(opt.model).to(device)      # create a model given opt.model and other options
    model.eval()
    # get input data
    if os.path.isdir(opt.dataroot):
        test_list = get_image_list(opt.dataroot)
    elif os.path.isfile(opt.dataroot):
        test_list = [opt.dataroot]
    else:
        raise Exception("{} is not a valid directory or image file.".format(opt.dataroot))
    # save outputs
    save_dir = opt.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    for test_path in tqdm(test_list):
        basename = os.path.basename(test_path)
        aus_path = os.path.join(save_dir, basename)
        img, aus_resize = read_img_path(test_path, opt.load_size)

        if opt.clahe_clip > 0:
            img = (img + 1) / 2 # [-1,1] to [0,1]
            img = equalize_clahe(img, clip_limit=opt.clahe_clip)
            img = (img - .5) / .5 # [0,1] to [-1,1]

        aus_tensor = model(img.to(device))
        print(aus_tensor.shape)
        aus_img = tensor_to_img(aus_tensor)
        save_image(aus_img, aus_path, aus_resize)
"""
