import os
import math
from random import randint
import cv2 as cv
import numpy as np

# 读取原始图像
for _, _, files in os.walk('out/result_1'):
    for file in files:
        if file.endswith('txt'):
            continue
        img = cv.imread(os.path.join('out/result_1', file))

        # 获取图像行和列
        print(img.shape)
        rows, cols = img.shape[:2]

        # 设置中心点和光照半径
        centerX = rows / 2 + randint(-5, 5)
        centerY = randint(0, cols)
        radius = randint(10, rows)

        # 设置光照强度
        strength = randint(100,200)

        # 新建目标图像
        dst = np.zeros((rows, cols, 3), dtype="uint8")

        # 图像光照特效
        for i in range(rows):
            for j in range(cols):
                # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
                distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
                # 获取原始图像
                B = img[i, j][0]
                G = img[i, j][1]
                R = img[i, j][2]
                if distance < radius * radius:
                    # 按照距离大小计算增强的光照值
                    result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                    B = img[i, j][0] + result
                    G = img[i, j][1] + result
                    R = img[i, j][2] + result
                    # print(type(B),type(img[i, j][0]))
                    # 判断边界 防止越界
                    B = min(255, max(0, B))
                    G = min(255, max(0, G))
                    R = min(255, max(0, R))
                    dst[i, j] = np.uint8((B, G, R))
                    # exit(0)

                else:
                    dst[i, j] = np.uint8((B, G, R))
        # cv.imwrite(os.path.join('./light/', file), dst)
        cv.imshow(file,dst)
        cv.waitKey()
