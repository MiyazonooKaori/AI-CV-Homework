import cv2
import random
import numpy as np
from matplotlib import  pyplot as plt

# "打开一张图片，0：灰度"
img = cv2.imread('./bleach.jpg',)
cv2.imshow('bleach_crop',img)

key = cv2.waitKey()
if key ==27 : #escape 退出
    cv2.destroyAllWindows()

#image crop 切片
img_crop = img[0:100,0:200]
cv2.imshow('bleach_crop',img_crop)

key = cv2.waitKey()
if key ==27 : #escape 退出
    cv2.destroyAllWindows()


# histogram 统计 直方图 color shift
print(img.shape[0]*0.5)
# 对图像进行缩放
img_small_brighter = cv2.resize(img,(int(img.shape[0]*0.5),int(img.shape[1]*0.5)))
print(img_small_brighter.shape)
# 直方图
plt.hist(img.flatten(),256,[0,256],color = 'r')
# 图像颜色的空间转换
img_yuv = cv2.cvtColor(img_small_brighter,cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])  # only for 1 channel
img_output = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR) # y:luminance(明亮度)，u&v：色度饱和度
cv2.imshow('Color input image',img_small_brighter)
cv2.imshow('Histogram equalized',img_output)
key = cv2.waitKey()
if key ==27 : #escape 退出
    cv2.destroyAllWindows()

# rotation
M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0] / 2),30,1) # center , angle , scale
img_rotate = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
cv2.imshow('rotated lenna',img_rotate)
key = cv2.waitKey()
if key ==27 : #escape 退出
    cv2.destroyAllWindows()

# perspective transform
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()