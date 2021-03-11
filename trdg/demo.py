import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gauss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image=cv2.imread(image)
    image = np.array(image/255, dtype=float)
    # print(image)
    print(image.shape)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    print(noise.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    cv2.imshow("gasuss", out)
    cv2.waitKey()
    # return out
# gauss_noise('images/containers.jpg')
# cv2.im(gauss_noise('images/containers.jpg'))
# import cv2  

# from PIL import Image  
# import numpy  
# image = Image.new("L", (679, 500))
# print(image.size)
from PIL import Image
import random as rnd
import math
def quasicrystal(width,height,image=None):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    # image = Image.new("L", (width, height))
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
def gauss_noise_customized(image, angle=90,mean=0, var=0.07):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    width,height=image.size
    img_arr=np.asarray(image)
    if img_arr.shape[2]==3:
        img_arr=cv2.cvtColor(img_arr,cv2.COLOR_RGB2RGBA)
    img_arr=img_arr/255
    # img_cv=np.transpose(img_cv,(1,0,2))
    img_arr=np.rot90(img_arr)
    ## Plan B
    noise = np.random.normal(mean, var,(width,height,4))
    out = img_arr + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return Image.fromarray(out,mode='RGBA').rotate(-90,expand=True)


# import matplotlib.pyplot as plt 
# mu, sigma = 0, 0.1 # mean and standard deviation s = np.random.normal(mu, sigma, 1000)
# s = np.random.normal(mu, sigma, 1000)
# count, bins, ignored = plt.hist(s, 30, density=True) 
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *  np.exp( - (bins - mu)**2 / (2 * sigma**2) ),  linewidth=2, color='r') 
# plt.show()

import numpy as np
import cv2
def gaussian(image, angle=0,mean=0, var=0.07):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    width,height=image.size
    img_arr=np.asarray(image)/255
    img_arr=np.transpose(img_arr, (1,0,2))
    print(img_arr.shape)
    ## Plan B
    noise = np.random.normal(mean, var,(width,height,4))
    print(noise.shape)
    out = img_arr + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    # out=np.transpose(out,(1,0,2))
    # return Image.fromarray(out,mode='RGBA').rotate(angle,expand=True)
    return Image.fromarray(out,mode='RGBA')

def sp_noise(image, prob=None):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob=random.uniform(0.01,0.05)
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs <prob] = black
    image[probs > 1 -prob] = white
    return Image.fromarray(image).convert("RGBA")
def shadow_light(image):
    # 获取图像行和列
    rows,cols=image.size
    # 设置中心点和光照半径
    centerX = cols / 2 + random.randint(-5, 5)
    centerY = random.randint(0, rows)
    radius = random.randint(10, cols)
    pixels=np.array(image)
    pixels=pixels/255 #paralle
    # 设置光照强度
    strength =  random.randint(-150,-130)/255
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
    return Image.fromarray(pixels).convert('RGBA')

    # dst=np.clip(dst,0,255)
    # Image.fromarray(dst).show()

    #         # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
    #         distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)

    #         if distance<radius*radius:
    #             result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
    #             # pixels[i][j][:3]+=result
    #             np.add(pixels[i][j][:3], result, out=pixels[i][j][:3], casting="unsafe")
    #         B,G,R=pixels[i,j][:3]
    #         # 判断边界 防止越界
    #         B = min(255, max(0, B))
    #         G = min(255, max(0, G))
    #         R = min(255, max(0, R))
    #         pixels[i, j][:3] = np.uint8((B, G, R))
    # pixels=np.clip(pixels,0,255)
    # print(pixels)
    # return Image.fromarray(pixels)

# image = Image.open("out/result_1/sp_9.jpg") 
image=Image.open('images/city.jpg') 
# shadow_light(image.convert('RGBA')).show()
# shadow_light(image.convert('RGBA'))
# # img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2RGBA)
# # print(img.shape)
# # print(image.mode)
# # image=sp_noise(img,0.03)
# # image=gauss_noise_customized(image.convert("RGBA"))
# img_rgba=image.convert('RGBA')
# print(np.arra)
# gauss_noise_customized(image).show()

import cv2
import numpy as np

# Some input images
img1 = cv2.resize(cv2.imread('images/city.jpg'), (400, 300))
img2 = cv2.resize(cv2.imread('images/yyzz/b3.png'), (400, 300))
# print(img1.shape)
img1=img1/255
img2=img2/255
mask_line=np.repeat(0.3,img1.shape[1])
# Generate blend masks, here: linear, horizontal fading from 1 to 0 and from 0 to 1
# mask1 = np.repeat(np.tile(np.linspace(0.1, 0.3, img1.shape[1]), (img1.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
mask1 = np.repeat(np.tile(mask_line, (img1.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
# mask2 = np.repeat(np.tile(np.linspace(0, 1, img2.shape[1]), (img2.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
# print(mask1.shape)

cv2.imshow('mask1', mask1)
final=img1*mask1+img2
final=np.clip(final,0,1)
final=np.uint8(final*255)
# cv2.imshow('mask2', mask2)
# # Generate output by linear blending
# final = np.uint8(img1 * mask1)

# # Outputs
# # cv2.imshow('img1', img1)
# # cv2.imshow('img2', img2)
# # cv2.imshow('mask1', mask1)
# # cv2.imshow('mask2', mask2)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()


# width,height=img_rgba.size
# print(width,height)
# # image.show()
# # noise=np.random.normal(0,0.1,(width,height,4))
# img_arr=np.asarray(img_rgba)
# img_arr=np.transpose(img_arr, (1,0,2))
# print(img_arr.shape)
# img_result=Image.fromarray(img_arr,mode='RGBA')
# img_result.show()
# img_rotate=img_result.rotate()
# img_rotate.show()

# print('fromarray',img_result.size)
# # img_result.show()
# img_cv=cv2.cvtColor(img_arr,cv2.COLOR_RGB2RGBA)
# print(img_cv.shape)
# img_result_cv=Image.fromarray(img_cv,mode='RGBA')
# print('fromarray',img_result_cv.size)

# result=np.zeros((width,height,4))
# for w in range(width):
#     for h in range(height):
#         result[w][h]=np.array(pixels[w,h])
# result=result/255
# result+=noise
        # # print(pixels[w,h],noise[w][h])
        # scaled_pixel=np.array(pixels[w,h])/255
        # # result.append(scaled_pixel)
        # # print(scaled_pixel)
        # result[w][h]=scaled_pixel+noise[w][h]
        # # print(result[w][h])
        # # exit(0)

# result = np.clip(result, 0, 1.0)
# result = np.uint8(result*255)
# print(result.shape)
# img_result=Image.fromarray(result).convert("RGBA")
# img_result.show()


# image.show()
# import os
# w_f_line=[]

# with open('texts/demo_all/text_1.txt','r',encoding='utf-8') as f:
#     total=f.read().splitlines()
#     for i in range(2):
#         with open('texts/demo_all/text_1_{}.txt'.format(str(i)),'w',encoding='utf-8') as w_f:
#             # w_f.write()
#             print(i)
#             try:
#                 for line in range(i*50000,(i+1)*50000):
#                     w_f.write(total[line]+'\n')
#             except IndexError:
#                 for each_line in total[line:-1]:
#                     w_f.write(each_line+'\n')
#                 break
# for file_index,file in w_f_line:
#     with open('','w',encoding='utf-8') as w_f:
#         for line in file:
#             w_f.write()
#             w_f.writelines
#                 f.write("{}-{}: {} {}\n".format(bg_img_name_list[i],fonts_list[i],file_name,strings[i]))

# with open('test.txt','w') as f:
#     f.writelines(['ok','hello'])

            