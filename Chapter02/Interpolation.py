import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image as IMG
import cv2 as cv
np.random.seed(5)

img = IMG.open(r"C:\Users\qhzhuang\Desktop\2.jpg")
img = np.asarray(img)
img = img[:, :, 0]
# img_0 = img[:, :, 0]
# img_1 = img[:, :, 1]
# img_2 = img[:, :, 2]
# print(img_r.dtype)
# print(img_0)
# img_gray = 0.299 * img_1 + 0.587 * img_0 + 0.114 * img_2
# print(img.dtype)
# img_g = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# matplotlib 以RGB显示


def interpolation(img, times):  # 传入一个(H*W)尺寸的原始图像img(ndarray对象)和放大倍数times
    ratio = 1 / times
    shape0 = math.floor(times * img.shape[0])
    shape1 = math.floor(times * img.shape[1])
    res = np.ones([shape0, shape1], dtype=np.uint8)
    for i in range(0,  shape0):
        for j in range(0, shape1):
            res[i][j] = img[math.floor(i * ratio)][math.floor(j * ratio)]
    return res


def generate_list(img, num):  # 生成含噪声的图像
    img_list = []
    shape = img.shape
    for i in range(0, num):
        img_list.append(img + np.random.randint(10, 100) * np.random.randn(shape[0], shape[1]))
    return img_list


def average(img_list):  # 传入待加和图像列表

    res = np.zeros(img_list[0].shape, dtype=img_list[0].dtype)
    for i in img_list:
        res += i
    return np.asarray(res / len(img_list), dtype=np.uint8)


img_list = generate_list(img, 100)

img_ip = interpolation(img, 1.5)
# print(img_ip)
# plt.imshow(img, cmap="gray")
# plt.title("original")
# plt.imshow(img_ip, cmap="gray")
# plt.title("interpolation")
fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(img, cmap="gray")
ax[0][0].set_title("original image")
ax[0][1].imshow(img_list[0], cmap="gray")
ax[0][1].set_title("image added noise")
ax[1][0].imshow(average(img_list[0:50]), cmap="gray")
ax[1][0].set_title("50 img average")
ax[1][1].imshow(average(img_list[0:]), cmap="gray")
ax[1][1].set_title("100 img average")
plt.show()
# opencv 显示图像以BGR模式
cv.imshow("Original image", img,)
cv.imshow("After Interpolation", img_ip)
cv.waitKey(0)
cv.destroyAllWindows()
# print(math.floor(3.33))