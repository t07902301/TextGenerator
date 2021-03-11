import cv2
import math
import os
import random as rnd
import numpy as np
import random

from PIL import Image, ImageDraw, ImageFilter
def add_noise(noise_type,image):
    if noise_type==1:
        image=gaussian_noise(image)
    elif noise_type==2:
        image=sp_noise(image)
    return image
def gaussian_noise(image,mean=0, var=0.07):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    width,height=image.size
    img_arr=np.asarray(image)
    if img_arr.shape[2]==3:# Expand the last dimension to 4 in alignment with the noise array
        img_arr=cv2.cvtColor(img_arr,cv2.COLOR_RGB2RGBA)
    img_arr=img_arr/255
    noise = np.random.normal(mean, var,(height,width,4))
    out = img_arr + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return Image.fromarray(out,mode='RGBA')
def sp_noise(image):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob=random.uniform(0.01,0.02)
    img_arr=np.array(image)
    if len(img_arr.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = img_arr.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(img_arr.shape[:2])
    img_arr[probs <prob] = black
    img_arr[probs > 1 -prob] = white
    return Image.fromarray(img_arr).convert("RGBA")
def shadow_light(image,degree):
    # 获取图像行和列
    rows,cols=image.size
    # 设置中心点和光照半径
    centerX = cols / 2 + random.randint(-5, 5)
    centerY = random.randint(0, rows)
    radius = random.randint(10, cols)
    pixels=np.array(image)
    if pixels.shape[2]==3:
        pixels=cv2.cvtColor(pixels,cv2.COLOR_RGB2RGBA)
    pixels=pixels/255 #paralle
    # 设置光照强度
    strength =  random.randint(-degree,degree)/255
    # 图像光照特效
    for i in range(cols):
        for j in range(rows):
            # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
            if distance < radius * radius:
                # 按照距离大小计算增强的光照值
                result = (strength * (1.0 - math.sqrt(distance) / radius))
                pixels[i][j][:3]+=result
    pixels=np.clip(pixels,0,1)
    pixels= np.uint8(pixels*255)
    return Image.fromarray(pixels,mode='RGBA')

def plain_white(height, width):
    """
        Create a plain white background
    """

    return Image.new("L", (width, height), 255).convert("RGBA")


def quasicrystal(height, width):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = rnd.random() * 30 + 20  # frequency
    phase = rnd.random() * 2 * math.pi  # phase
    rotation_count = rnd.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return image.convert("RGBA")


def add_background(height, width, images,index):
    """
        Create a background with a image
    """
    img_index=index%len(images)# or random
    pic = Image.open(
        images[img_index]
    )
    if pic.size[0] < width:
        pic = pic.resize(
            [width, int(pic.size[1] * (width / pic.size[0]))], Image.ANTIALIAS
        )
    if pic.size[1] < height:
        pic = pic.resize(
            [int(pic.size[0] * (height / pic.size[1])), height], Image.ANTIALIAS
        )

    if pic.size[0] == width:
        x = 0
    else:
        x = rnd.randint(0, pic.size[0] - width)
    if pic.size[1] == height:
        y = 0
    else:
        y = rnd.randint(0, pic.size[1] - height)

    return pic.crop((x, y, x + width, y + height))
def apply_gauss_blur(img, ks=None):
    if ks is None:
        ks = [7, 9, 11, 13]
    ksize = random.choice(ks)

    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize <= 3:
        sigma = random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img
def gen_rand_bg(height, width, is_bgr=True):
    """
    Generate random background
    """
    bg_high = random.uniform(220, 255)
    bg_low = bg_high - random.uniform(1, 60)

    bg = np.random.randint(bg_low, bg_high, (height, width)).astype(np.uint8)

    bg = apply_gauss_blur(bg)

    if is_bgr:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(bg).convert("RGBA")