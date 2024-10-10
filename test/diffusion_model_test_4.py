#diffusion_model_test_4.py
#目標
#1.添加躁聲且過程可視化

num_shows = 20 #顯示20個過程
fig,axs = plt.subplots(2,10,figsize=(28,3)) #前面那個畫圖的fig跟axs (x,y,z) 2行,10列.size
plt.rc('text',color='black') #文字黑

#共有10000個點，每個點包含兩個座標
#生成100步，每隔5步一個圖像
for i in range(num_shows):
    j = i//10 #第x張圖像的對應行數 ex loop到16張 j=1
    k = i%10 #列數 k=6
    q_i = q_x(dataset,torch.tensor([i*num_steps//num_shows])) #調用diffusion_model_test_3.py的q_x函數  #生成t時的樣本

    #dataset(10000,2),torch.tensor()=第x步的張量,[i*num_steps//num_shows]=計算我要顯示的步數 ex=第5張 i=5*100//20 ==>會是第25步

    #畫圖瞜
    axs[j,k].scatter(q_i[:,0],q_i[:,1],color='red',edgecolor='white')
    axs[j,k].set_axis_off()
    axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')