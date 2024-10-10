#diffusion_model_test_3.py
#目標
#1.Forward process的邏輯

def q_x(x_0,t):
    """可以用x[0]得到任意t的x[t]"""
    # 計算給定時間t的樣本x[t]

    #表示從原始數據 𝑥0 根據時間步𝑡 生成新的樣本 𝑥[𝑡]的過程。
    # x_0：dataset，原始(10000, 2)的數據
    # t: torch.tensor([i]),i為採樣了幾次

    noise = torch.randn_like(x_0) #生成與原始數據形狀相同的隨機躁聲 就是ppt的 ε~N(0,1)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)
    # x[t] = √αt⋅x0 + √1−αt ⋅ ε(noise)
    # 跟李教授的影片一樣