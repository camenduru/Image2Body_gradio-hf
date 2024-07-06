import cv2
import numpy as np
from PIL import Image
from anime import generate_sketch

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def scribble_xdog(img, res=512, thr_a=32, **kwargs):
    """
    XDoGを使ってスケッチ画像を生成する
    :param img: np.ndarray, 入力画像
    :param res: int, 出力画像の解像度
    :param thr_a: int, 閾値

    Returns
    -------
    Image : PIL.Image
    """
    img, remove_pad = resize_image_with_pad(img, res)
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(img, dtype=np.uint8)
    result[2 * (255 - dog) > thr_a] = 255
    result = Image.fromarray(remove_pad(result))
    return result, True

def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    result = cv2.Canny(img, thr_a, thr_b)
    result = Image.fromarray(remove_pad(result))
    return result, True

def get_sketch(image, method='scribble_xdog', res=2048, thr=20, **kwargs):
    # image: np.ndarray
    input_height = image.shape[0]
    input_width = image.shape[1]

    if method == 'scribble_xdog':
        processed_image, _ = scribble_xdog(image, res, thr) # PIL.Image
        processed_image = processed_image.resize((input_width, input_height))
        # make PIL.Image to cv2 and INVERSE
        processed_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        processed_image = 255 - processed_image
        processed_image = Image.fromarray(processed_image)
    elif method == 'anime2sketch':
        clahe = 1.0
        processed_image = generate_sketch(image, clahe_clip=clahe, load_size=1024) # output: numpy.ndarray
        processed_image = Image.fromarray(processed_image)
        # processed_image.save(output_path.split('.')[0] + f'_{clahe}.png')
    elif method == 'both':
        alpha = 0.5
        # 2枚をalphaの重みで合成
        scribble_xdog_processed_image, _ = scribble_xdog(image, res, thr)
        scribble_xdog_processed_image = scribble_xdog_processed_image.resize((input_width, input_height))
        scribble_xdog_processed_image = cv2.cvtColor(np.array(scribble_xdog_processed_image), cv2.COLOR_RGB2BGR)
        scribble_xdog_processed_image = 255 - scribble_xdog_processed_image

        anime2sketch_processed_image = generate_sketch(image, clahe_clip=1.0, load_size=1024)
        anime2sketch_processed_image = Image.fromarray(anime2sketch_processed_image)
        anime2sketch_processed_image = anime2sketch_processed_image.resize((input_width, input_height))
        anime2sketch_processed_image = cv2.cvtColor(np.array(anime2sketch_processed_image), cv2.COLOR_RGB2BGR)
        
        processed_image = cv2.addWeighted(scribble_xdog_processed_image, alpha, anime2sketch_processed_image, 1-alpha, 0)
        processed_image = Image.fromarray(processed_image)

    return processed_image
