import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image as IMG
import cv2 as cv
from functools import reduce
from collections import Counter

np.random.seed(5)

img = IMG.open(r"C:\Users\qhzhuang\Desktop\8.jpg")
img = np.asarray(img, dtype=np.uint8)

# img = img[:, :, 0]
L = 256


# img2 = img

# img2 = IMG.open(r"C:\Users\qhzhuang\Desktop\4.jpg")
# img2 = np.asarray(img2, dtype=np.uint8)
# img2 = img2[:, :, 0]


def get_histogram(img):
    histo = [0 for k in range(0, L)]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            histo[img[i][j]] += 1
    return histo


# def global_(img):
#     histo = get_histogram(img)
#     histo = np.asarray(histo)
#     return histo.mean(), histo.var()


# def local_(img, x, y, k_size):
#     sxy = int(k_size / 2)
#     x_lb = x - sxy
#     x_rb = x + sxy
#     y_ub = y - sxy
#     y_db = y + sxy
#     while x_lb < 0:
#         x_lb += 1
#     while x_rb > img.shape[0] - 1:
#         x_rb -= 1
#     while y_db > img.shape[1] - 1:
#         y_db -= 1
#     while y_ub < 0:
#         y_ub += 1
#     img_new = img[x_lb:x_rb + 1, y_ub: y_db + 1]
#     return img_new.mean(), img_new.var()


def image_negative(r):
    return L - 1 - r


def log_trans(r, c):
    return c * np.log(1.0 + r)


def gamma(r, c, gamma):
    return c * r ** gamma


def piecewise_trans(r, r1, s1, r2, s2):
    def proc_list(i):
        if 0 <= i < r1:
            return s1 / r1 * i
        elif r1 <= i <= r2:
            return (s2 - s1) / (r2 - r1) * (i - r2) + s1
        else:
            return (L - 1 - s2) / (L - 1 - r2) * (i - r2) + s2
    return [proc_list(j) for j in r]


def intensity_transformation(image, fn, **kwargs):
    return np.asarray(list(map(lambda x: fn(x, **kwargs), image)))


# test intensity trans
# fig, axes = plt.subplots(1, 2)
# trans_img = intensity_transformation(img, image_negative)
# fig.suptitle("Image Negative")

# trans_img = intensity_transformation(img, gamma, c=1, gamma=0.54)
# fig.suptitle("Gamma Transformation")

# fig.suptitle("Log Transformation")
# trans_img = intensity_transformation(img, log_trans, c=1)
#
# fig.suptitle("Piecewise-Linear Transformations")
# trans_img = intensity_transformation(img, piecewise_trans, r1=120, s1=100, r2=170, s2=200)
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title("Original image")
#
# axes[1].imshow(trans_img, cmap='gray')
# axes[1].set_title("Transformed image")


# def intensity_enhancement(img, E, k0, k1, k2, k_size):
#     new_img = np.zeros_like(img)
#     mean_glo, var_glo = global_(img)
#     for i in range(0, img.shape[0]):
#         for j in range(0, img.shape[1]):
#             mean_local, var_local = local_(img, i, j, k_size)
#             if mean_local <= k0 * mean_glo and k1 * var_glo <= var_local <= k2 * var_glo:
#                 new_img[i][j] = E * img[i][j]
#                 print("executed")
#             else:
#                 new_img[i][j] = img[i][j]
#
#     return new_img


# filtering


class GaussianSmooth:
    def __init__(self, radius=1, sigma=1.5):
        self.radius = radius
        self.sigma = sigma

    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigma * self.sigma)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigma * self.sigma))
        return res1 * res2

    def template(self): # 计算出掩模
        side_length = self.radius * 2 + 1
        result = np.zeros((side_length, side_length))
        result = np.array([[self.calc(i - self.radius, j - self.radius) for j in range(side_length)]
                           for i in range(side_length)])
        total = result.sum()
        return result / total  # 归一化

    def filt(self, image, template):
        arr = np.array(image)
        lenx = arr.shape[0]
        leny = arr.shape[1]
        padding = np.zeros((lenx + 2 * self.radius, leny + 2 * self.radius))  # 填充0保持大小一致
        padding[self.radius:self.radius + lenx, self.radius:self.radius + leny] = image
        # 对应元素相乘
        new_img = [[(template * padding[i-self.radius:i + self.radius + 1, j-self.radius:j + self.radius + 1]).sum()
                    for j in range(self.radius, leny + self.radius)] for i in range(self.radius, lenx + self.radius)]
        return new_img
#
#
#
# fig, axes = plt.subplots(1, 2)
# fig.suptitle("Gaussian Smooth")
# fig.set_tight_layout(True )
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title("Original image")
#
# axes[1].imshow(image2, cmap='gray')
# axes[1].set_title("Gaussian smooth")
# plt.show()


# filtering
class OrderStat:
    def __init__(self, cls, radius):
        self.radius = radius
        self.cls = cls

    def template(self):
        side_length = self.radius * 2 + 1
        result = np.ones((side_length, side_length))
        return result

    def filt(self, image, template):
        lenx = image.shape[0]
        leny = image.shape[1]
        padding = np.zeros((lenx + 2 * self.radius, leny + 2 * self.radius))
        padding[self.radius:self.radius + lenx, self.radius:self.radius + leny] = image
        if self.cls == 'mean':
            new_img = \
                [[(template * padding[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1])\
                .mean()
                  for j in range(self.radius, leny + self.radius)]
                 for i in range(self.radius, lenx + self.radius)
                 ]
        elif self.cls == 'max':
            new_img = \
                [[(template * padding[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]).max()
                  for j in range(self.radius, leny + self.radius)]
                 for i in range(self.radius, lenx + self.radius)
                 ]
        elif self.cls == 'min':
            new_img = \
                [[(template * padding[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]).min()
                  for j in range(self.radius, leny + self.radius)]
                 for i in range(self.radius, lenx + self.radius)
                 ]
        else:
            raise ValueError("only support min, max, mean filter")
        return new_img

#
# GBlur = OrderStat('mean', 7)
# temp = GBlur.template()
# image2 = GBlur.filt(img, temp)  # 高斯模糊滤波，得到新的图片
# fig, axes = plt.subplots(1, 2)
# fig.suptitle("Filtered with 7*7 mean filter)")
# fig.set_tight_layout(True)
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title("Original image")
#
# axes[1].imshow(image2, cmap='gray')
# axes[1].set_title("Filtered image")
# plt.show()


class Sharpening():
    def __init__(self, radius=1):
        self.radius = radius

    def template(self):

        return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int32)

    def filt(self, image, template):
        lenx = image.shape[0]
        leny = image.shape[1]
        padding = np.zeros((lenx + 2 * self.radius, leny + 2 * self.radius))
        padding[self.radius:self.radius + lenx, self.radius:self.radius + leny] = image
        new_img = [[(template * padding[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]).sum()
                  for j in range(self.radius, leny + self.radius)]
                 for i in range(self.radius, lenx + self.radius)
                 ]
        return new_img

GBlur = Sharpening()
temp = GBlur.template()

image2 = GBlur.filt(img, temp)  # 高斯模糊滤波，得到新的图片


fig, axes = plt.subplots(1, 2)
fig.suptitle("Laplacian sharpening)")
fig.set_tight_layout(True)
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original image")

axes[1].imshow(image2, cmap='gray')
axes[1].set_title("Filtered image")
plt.show()



def equalization(img):
    freq = get_histogram(img)
    equalized_img = np.zeros_like(img, dtype=np.uint8)
    new_freq = [0 for i in range(0, 256)]
    dic = {}
    for r in range(0, 256):
        s = int((L - 1) * reduce(lambda x, y: x + y, freq[0:r + 1]) / (img.size))
        if freq[r] > 0:
            dic[r] = s
        new_freq[s] = freq[r]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            equalized_img[i][j] = dic[img[i][j]]
    return new_freq, equalized_img, dic


def matching(img, desired_img):
    matching_img = np.zeros_like(img, dtype=np.uint8)
    _, __, r_s = equalization(img)
    _, __, z_s = equalization(desired_img)
    s_z = {v: k for k, v in z_s.items()}
    print(r_s)
    print(z_s)
    print(s_z)
    dic_final = {}
    for r in r_s.keys():
        s = r_s[r]
        if s not in s_z.keys():
            # s1 = s
            s2 = s
            # while s1 not in s_z.keys():
            #     s1 = s1 - 1
            while s2 not in s_z.keys():
                s2 = s2 + 1
            s = s2
        dic_final[r] = s_z[s]

        # = {i: s_z[r_s[i]] for i in r_s.keys() }
    print(dic_final)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            matching_img[i][j] = dic_final[img[i][j]]
    return matching_img


# test equalizztion

# his1 = get_histogram(img)
# his2, img2, dic = equalization(img)
# fig, axes = plt.subplots(2, 2)
# axes[0, 0].imshow(img, cmap='gray')
# axes[0, 0].set_title("Original image")
#
# axes[0, 1].bar(x=list(range(0, 256)), height=his1)
# axes[0, 1].set_title("Histogram of original image")
#
# axes[1, 0].imshow(img2, cmap='gray')
# axes[1, 0].set_title("Processed image")
#
# axes[1, 1].bar(x=list(range(0, 256)), height=his2)
# axes[1, 1].set_title("Histogram of processed image")
# fig.suptitle("Equalization")
#

print('----------------------------')
# test matching


# his1 = get_histogram(img)
# his2 = get_histogram(img2)
# img_processed = matching(img, img2)
# his_proc = get_histogram(img_processed)
# fig, axes = plt.subplots(3, 2)
# axes[0, 0].imshow(img, cmap='gray')
# axes[0, 0].set_title("Original image")
#
# axes[0, 1].bar(x=list(range(0, 256)), height=his1)
# axes[0, 1].set_title("Histogram of original image")
#
# axes[1, 0].imshow(img2, cmap='gray')
# axes[1, 0].set_title("Desired image")
#
# axes[1, 1].bar(x=list(range(0, 256)), height=his2)
# axes[1, 1].set_title("Histogram of Desired image")
#
# axes[2, 0].imshow(img_processed, cmap='gray')
# axes[2, 0].set_title("Processed image")
#
# axes[2, 1].bar(x=list(range(0, 256)), height=his_proc)
# axes[2, 1].set_title("Histogram of Processed image")
# fig.suptitle("Image matching")
#
# plt.show()



# img2 = cv.GaussianBlur(img, (7, 7), 0)
# cv.imshow("lap", image2)
cv.imshow("Original image", img)
img2 = cv.Laplacian(img, cv.CV_8U)
cv.imshow("Equalized image", img2)
# cv.imshow("After Interpolation", img_ip)
cv.waitKey(0)
cv.destroyAllWindows()
