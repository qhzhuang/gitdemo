# import numpy as np
# import math
# import matplotlib
# import matplotlib.pyplot as plt
# from PIL import Image as IMG
# from Chapter02.chap_03 import get_histogram
# import cv2 as cv
# from functools import reduce
# from collections import Counter
# # matplotlib.use('Agg')
# np.random.seed(5)
#
# img = IMG.open(r"C:\Users\qhzhuang\Desktop\2.jpg")
# img = np.asarray(img, dtype=np.uint8)
# img = img[:, :, 0]
#
# zero = np.zeros((3, 5), dtype=np.int32)
#
# def fn(i, c):
#     return i + c
#
# new_zero = map(fn, zero)
#
# print(np.array(list(new_zero)))
import cv2 as cv
import numpy as np
img = cv.imread(r"C:\Users\qhzhuang\Desktop\8.jpg")

cv.imshow('srcImage', img)
cv.imshow('grayImage', grayimg)
cv.imshow("Laplacian",grayimg+shp[1:shp.shape[0]-1,1:shp.shape[1]-1])
cv.waitKey(0)
cv.destroyAllWindow()