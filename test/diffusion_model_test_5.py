#diffusion_model_test_5.py
#目標
#1. Reverse process 啟動

import torch
import torch.nn as nn

#一個MLP 應該是用於denoise
class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=128): #初始化
        super(MLPDiffusion,self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(2,num_units), #2維到128維的全連接
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,2), #128維回到2維
            ]
        )
        self.step_embeddings = nn.ModuleList( #embedding 三個嵌入層 #nn.modulelist一種 PyTorch 容器 
            [
                nn.Embedding(n_steps,num_units), #nn.embedding(100,128) 將100個時間步丟入128維中 #https://ithelp.ithome.com.tw/articles/10222044
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]#使得模型能夠根據時間步的不同獲得可學習的嵌入特徵，這樣有助於增強模型在擴散過程中的表現。
        )
    def forward(self,x,t): #forward propagation
        #  x = x_0
        for idx,embedding_layer in enumerate(self.step_embeddings): 
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)
        
        return x
    

# nn.Embedding：https://blog.csdn.net/qq_39540454/article/details/115215056
# 使用範例
# model = MLPDiffusion(num_steps)
# output = model(x,step)