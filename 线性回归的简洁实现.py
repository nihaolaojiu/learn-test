#!/usr/bin/env python
# coding: utf-8

# In[1]:


#生成数据集
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
true_w = torch.tensor([2,-3.4]).float()
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)
#读取数据集
def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
batch_size = 10
data_iter = load_array((features,labels),batch_size)
next(iter(data_iter))
#定义模型
from torch import nn#nn是神经网络的缩写
net=nn.Sequential(nn.Linear(2,1))
#PyTorch库中的nn.Sequential函数定义了一个神经网络模型。nn.Linear（2， 1）表示了一个线性层，输入维度为2，输出维度为1。
#这意味着该神经网络模型有2个输入特征和1个输出。
#初始化模型参数
net[0].weight.data.normal_(0,0.01)#使用正态分布的随机数，均值为0，标准差为0.01，来初始化权重值
net[0].bias.data.fill_(0)#将网络模型的第一个层的偏置参数全部设置为0.
#定义损失函数
loss = nn.MSELoss()#均方误差（MSE）
#定义优化算法
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
#使用随机梯度下降（SGD）优化器来更新神经网络中的参数。
#其中，net.parameters（）返回了神经网络中所有需要优化的参数，lr参数表示学习率，即每次更新的步长大小。
#训练
num_epochs = 3#数据扫三遍，三次回归运算
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X) ,y)
        trainer.zero_grad()
        trainer.step()
    l=loss(net(features),labels)
    print(f'epoch{epoch+1},loss {l:f}')
w=net[0].weight.data
print('w的估计误差：',true_w-w.reshape(true_w.shape))
b=net[0].bias.data
print('b的估计误差：',true_b-b)


# In[ ]:




