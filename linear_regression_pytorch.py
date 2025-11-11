import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
data = pd.read_csv('./data/income.csv')
plt.scatter(data.Education, data.Income)
plt.xlabel('Education')
plt.ylabel('Income')
# plt.show()  # 把散点图展示出来

# 分解写法 wx+b，线性公式，反向传播时候w，b要求导
w = torch.randn(1, requires_grad=True)  # 初始化的时候小一点，w必须作为叶子节点，不可以*0.02
b = torch.zeros(1, requires_grad=True)  # b初始化为0

learning_rate = 0.001

# 如果不去做reshape，那Y是1维， Y中的y就是标量，是0维
# reshape(-1, 1) = "你帮我算行数，列数固定为1"
# reshape(1, -1) = "行数固定为1，你帮我算列数"

X = torch.from_numpy(data.Education.values.reshape(-1, 1)).type(torch.FloatTensor)
Y = torch.from_numpy(data.Income.values.reshape(-1, 1)).type(torch.FloatTensor)

# 定义训练过程
for epoch in range(1):  # 训练一次就是一个epoch,一般执行5000次，为了测试后面的代码，临时写为1
    for x, y in zip(X, Y):
        y_pred = torch.matmul(x, w) + b
        # 损失函数
        loss = (y - y_pred).pow(2).sum()  # 损失函数平方求和
        # pytorch 对一个变量多次求导，求导结果会累加起来
        if w.grad is not None:
            # 重置 w的导数
            w.grad.data.zero_()  # 第二次进来时候，包含上一次的导函数结果，调用划线方法，修改原值
        if b.grad is not None:
            b.grad.data.zero_()
        # 反向传播，求w，b的导数
        loss.backward()
        # 梯度下降，更新w,b
        with torch.no_grad():
            w.data -= learning_rate * w.grad.data
            b.data -= learning_rate * b.grad.data

print(w)
print(b)
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), (torch.matmul(X, w) + b).data.numpy(), c='r')
# plt.show()

## pytorch 封装写法

from torch import nn

# nn 就是神经网络
# nn.Linear 和tensorflow 中的dense是一个意思，wx+b
model = nn.Linear(in_features=1, out_features=1, bias=True)  # 相当于一个线性运算
# in_features 输入维度，out_features输出的维度是多少，它会把w的维度算出来
# bias参数表示是否在线性变换中添加偏置项。
# 输入是一个维度，输出也是一个维度

# 定义损失函数，均方差
loss_fn = nn.MSELoss()

## 定义优化器
# 优化器的第一个参数必须是要更新的模型中的参数
# lr 学习率 参数必须写， momentum 动量 一般设置为0.9
opt = torch.optim.SGD(model.parameters(), lr=0.001)

# 训练
for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = model(x)
        loss = loss_fn(y, y_pred)

        # 梯度清零操作
        opt.zero_grad()
        loss.backward()  # 执行反向传播
        # 更新 操作
        opt.step()  # 代码就固定了
print('*' * 20)
print(model.weight)
print(model.bias)
