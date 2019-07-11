import numpy as np
import cv2
def medianBlur(img,kernel,padding_way):
    img_h = int(img.shape[0])
    img_w = int(img.shape[1])

    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
# 获取填充后的size
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_w = int(img_w + 2 * padding_w)
# 选择填充方式
    if padding_way == "ZERO":
        img_padding = np.zeros((convolve_h,convolve_w))
        img_padding[padding_h:padding_h+img_h,padding_w:padding_w+img_w] = img[:,:]
    elif padding_way == "REPLICA":
        img_padding = np.zeros((padding_h,convolve_w))

    img_convolve = np.zeros(img.shape)

    for i in range(padding_h, padding_h + img_h):
            for j in range(padding_w, padding_w + img_w):
                # 中值滤波
                img_padding[padding_h,padding_w]=mid_filter(img_padding[i - padding_h:i + padding_h+1, j - padding_w:j + padding_w+1],kernel_h,kernel_w)
                # 卷积
                img_convolve[i - padding_h][j - padding_w] = int(np.sum(img_padding[i - padding_h:i + padding_h+1, j - padding_w:j + padding_w+1]*kernel))

    return img_convolve

def mid_filter(image,image_h,image_w):
    image_s=[]
    for i in range(0,image_h):
        for j in range(0,image_w):
            image_s.append(image[i,j])
    image_s.sort()

    return image_s[int(image_h*image_w/2)+1]

"""
img = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])
"""
kernel = np.array([[1,1,1],
                  [1,2,1],
                  [1,1,1]])

img=cv2.imread('D:/AI/lena.jpg',0)
cv2.imshow('ww',img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(medianBlur(img,kernel,"ZERO"))
cv2.imshow('ee',medianBlur(img,kernel,"ZERO"))
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()