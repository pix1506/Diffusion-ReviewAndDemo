#diffusion_model_test_6.py
#目標
#denoise訓練時的誤差函數?

def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    batch_size = x_0.shape[0]
    

    t = torch.randint(0,n_steps,size=(batch_size//2,))
    t = torch.cat([t,n_steps-1-t],dim=0)
    t = t.unsqueeze(-1)

    a = alphas_bar_sqrt[t] # torch.Size([batchsize, 1])
    

    aml = one_minus_alphas_bar_sqrt[t] # torch.Size([batchsize, 1])

    e = torch.randn_like(x_0) # torch.Size([batchsize, 2])
    

    x = x_0*a+e*aml # torch.Size([batchsize, 2])
    

    output = model(x,t.squeeze(-1)) #t.squeeze(-1)為torch.Size([batchsize])
    # output:torch.Size([batchsize, 2])

    return (e - output).square().mean()