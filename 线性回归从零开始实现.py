#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
"""设置inline的方式，直接把图片画在网页上"""
import random 
"""导入随机函数库"""
import torch
from d2l import torch as d2l
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    """定义正态分布"""
    y = torch.matmul(X, w) + b
    """矩阵乘积"""
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 10
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:',features[0],'\nlabel:',labels[0])
d2l.set_figsize()
"""设置图形大小"""
d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1);"""绘制散点图"""
def data_iter(batch_size,features,labels):#定义一个迭代数据
    num_examples=len(features)#样本数量n（num_example)
    indices = list(range(num_examples))#转成list
    random.shuffle(indices)#random.shuffle():将一个列表中的元素打乱，使之可以随机访问。
    for i in range(0,num_examples,batch_size):#从0-n，间隔一个batch_size
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])#截取切片 开始位置为i 结束位置为min函数的返回值 返回值为i+batch_size和num_examples的值比较小的那个
        yield features[batch_indices],labels[batch_indices]
batch_size=10
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)#requires_grad=True:用于指定一个张量是否需要计算梯度
b=torch.zeros(1,requires_grad=True)
def linreg(x,w,b):
    return torch.matmul(x,w)+b
def squared_loss(y_hat,y):#定义均方损失
    return (y_hat-y.reshape(y_hat.shape))**2/2
def sgd(params,lr,batch_size):#定义随机梯度下降；params包含所有参数（list），lr为学习率，batch_size为批量大小
    with torch.no_grad():#不需要计算梯度
        for param in params:
            param -= lr*param.grad/batch_size#学习率*梯度/batch_size
            param.grad.zero_()#梯度设置为0，为下一次计算
lr=0.03#学习率
num_epochs = 3#数据扫三遍，三次回归运算
net = linreg
loss = squared_loss#均方损失
for epoch in range(num_epochs):#进行三次训练
    for X,y in data_iter(batch_size,features,labels):#随机取出一定量（10个）样本到xy中
        l = loss(net(X,w,b),y)#net（也就是linreg）返回计算出的y的值，通过loss（squared_loss）计算损失，保存到长为批量大小的向量l中
        l.sum().backward()#算梯度（求导）
        sgd([w,b],lr,batch_size)
    with torch.no_grad():#计算下损失，不需要计算梯度
        train_l = loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')
print(f'w的估计误差:{true_w-w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b-b}')



# In[ ]:




