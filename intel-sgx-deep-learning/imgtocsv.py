import cv2
import os
import numpy as np
import pandas as pd 

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

c=get_imlist(r"./")
d=len(c)    # 图像个数
# 遍历每张图片

for i in range(d):
    img = cv2.imread(c[i],cv2.IMREAD_GRAYSCALE)  # 打开图像
    img_ndarray = np.asarray(img, dtype='float64') / 256  # 将图像转化为数组并将像素转化到0-1之间
    data = cv2.resize(img_ndarray,(128,128))
    data = np.resize(data,(1,128*128))
     
save = pd.DataFrame(data)
save.to_csv('./test.csv', index=True, header=False,sep=',') 
