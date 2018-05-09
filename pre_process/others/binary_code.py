# coding=utf-8
from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage import io,color
import skimage.morphology as sm
import matplotlib.pyplot as plt
# 以灰度模式读取图像
img = cv2.imread("22.jpeg", 0)
# 设置阈值进行二值化
# 注意这里二值化的同时对图像进行了反色，因为背景比前景颜色要浅，
# 直接二值化的结果是背景是白色的，前景是黑色的，这显然不是我们想要的结果
# 同时注意这样两个操作相加的这种写法
ret, binary = cv2.threshold(img, 0, 155, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

# 调用距离变换函数
# 第一个参数是二值化图像
# 第二个参数是类型distanceType
# 第三个参数是maskSize
# 返回的结果是一张灰度图像，但注意这个图像直接采用OpenCV的imshow显示是有问题的
# 所以采用Matplotlib的imshow显示或者对其进行归一化再用OpenCV显示
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# 将灰度、二值化图像合并到一个窗口中显示
dst=sm.closing(binary,sm.disk(10))
result = np.hstack((img, dst))

cv2.imshow("result", result)
# plt.imshow(dist, cmap='gray')
# plt.show()
cv2.waitKey(0)