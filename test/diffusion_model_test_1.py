#diffusion_model_test_1.py
#目標
#1.先利用s_curve先生成一個三維s
#2.轉換成二維並畫出平面的s
#3.轉換成二維張量

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
from mpl_toolkits.mplot3d import Axes3D 
import torch

s_curve,_ = make_s_curve(10**4,noise=0.1) # 生成10000個點的S曲線，噪聲強度0.1
s_curve = s_curve[:,[0,2]]/10.0 #只取第1跟第3維度，並除以10(/10)調整比例方便閱讀

print("shape of s:",np.shape(s_curve)) #印出數據s_curve的形狀 (只有二維，10000個點) (10000,2)

data = s_curve.T #.T 是Numpy的轉置操作將行變成列，列變成行。
#EX:原本 s_curve 是一個 (10000, 3) 的矩陣，轉置後變成 (3, 10000)，用意是每一列代表 X、Y 和 Z 軸的坐標，方便在後續的繪圖操作中使用。
#(2,10000)

fig,ax = plt.subplots() #plt.subplots()是matplotlib的函數 設定抓出整個圖形(fig)跟子圖形(ax)來設定

ax.scatter(*data,color='blue',edgecolor='white'); #*data是一種是將參數分離的語法，轉置過的data設定數據點顏色藍色，邊緣白的
#scatter可以繪製散點圖

ax.axis('off')#隱藏坐標軸

dataset = torch.Tensor(s_curve).float()#把s_curve變成張量，資料型別float 二維張量 用於後續操作
'''
標量（0維張量）：
僅有一個數值，例如 5 或 3.14。這是最基本的張量。

向量（1維張量）：
由一組數值組成的列表，例如 [1, 2, 3]。這可以表示一個一維的數據集合。

矩陣（2維張量）：
由多個向量組成的表格（數據結構），例如：
[[1, 2, 3],
 [4, 5, 6]]
這表示一個包含兩行三列的數據結構。

高維張量（3維及以上）：
當數據的維度大於 2 時，可以使用張量來表示，例如：
3維張量可以想像成一個立方體（例如，圖片的 RGB 數據）。
4維張量可以用來表示一批圖片（例如，批量的圖片數據）。
'''
plt.show() 

#顯示圖形