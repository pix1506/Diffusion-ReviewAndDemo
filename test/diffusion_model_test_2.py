#diffusion_model_test_2.py
#目標
#1.設置參數及初始化

num_steps = 100 #設置步數100

#設定β的值
#β 是每一步添加的噪聲強度
betas = torch.linspace(-6,6,num_steps) #我也不知道為啥是-6~6，就隨機向量分布而已ㄅ，然後有100步
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5 #把Betas sigmoid(0~1)，然後*(0.5e-2 - 1e-5)+1e-5 
#(0.5e-2 - 1e-5)約為0.04，用意是把sigmoid的輸出控制再這個範圍(0~0.04)，然後加一個常數避免有0

#設定α的值 
#α 值代表的是在每一步中保留原始數據的部分，而 β 是添加的噪聲強度。因此，α 可以視為數據中保留的成分
alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0) #torch.cumprod 函數計算沿著指定維度的累積乘積。這意味著每個步驟的 α 值將與之前所有步驟的 α 值相乘，以得到到目前為止的總保留成分。
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0) #再alphas_prod張量最前面放置一個1
'''
ex
Alphas: tensor([0.9999, 0.9998, 0.9995, 0.9991, 0.9985, 0.9985, 0.9991, 0.9995, 0.9998, 0.9999])
Alphas Product: tensor([0.9999, 0.9997(0.9999*0.9998), 0.9992, 0.9983, 0.9968, 0.9968, 0.9976, 0.9985, 0.9993, 0.9999])
Alphas Product P: tensor([1.0000, 0.9999, 0.9997, 0.9992, 0.9983, 0.9968, 0.9968, 0.9976, 0.9985, 0.9993])
'''

alphas_bar_sqrt = torch.sqrt(alphas_prod) #算alphas_prod的平方根
one_minus_alphas_bar_log = torch.log(1 - alphas_prod) #torch的log默認是loge
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
#擴散程度的指標?

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape 
#assert 語句：這是一個用於檢查條件的語句。
#如果條件為 False，則會引發 AssertionError，並停止程式的執行。這樣的語句常用於檢查代碼中的假設或邊界條件。
'''
這行代碼確認以下幾個張量的形狀是否一致：
alphas
alphas_prod
alphas_prod_p
alphas_bar_sqrt
one_minus_alphas_bar_log
one_minus_alphas_bar_sqrt
'''
print("all the same shape",betas.shape)
#ex: all the same shape torch.Size([100])
