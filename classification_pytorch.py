import numpy as np
import pandas as pd
import torch
from torch import nn

data = pd.read_csv('./data/finance_cards_data.csv')
print(data.head())
# 最后一列是标记，
print(data.shape)
X_data = data.iloc[:, 1:-1]  # 行都要，列第一行为id 不要，最后一列不要
print(X_data.shape)
Y_data = data.iloc[:, -1]  # 直接取最后一列
# series是不能作为标记的，把标记变成0 和1，方便最后求概率
print(Y_data.value_counts())
X = torch.from_numpy(X_data.values).type(torch.FloatTensor)
Y = torch.from_numpy(Y_data.values.reshape(-1, 1)).type(torch.FloatTensor)

# 回归和分类，区别不太大，回归后面加 一层sigmoid 非线性转换，就变成了分类

model = nn.Sequential(
    nn.Linear(in_features=X.shape[1], out_features=1),
    nn.Sigmoid()
)  # Sequentia 创建模型最简单的方式，一层输出层，一层sigmoid
#
# model = nn.Sequential(
#     nn.Linear(in_features=X.shape[1], out_features=1024),
#     nn.Linear(in_features=1024, out_features=1),
#     nn.Sigmoid()
# )  # 多层神经元的写法，效果没有更好
# BCE binary cross entropy 二分类交叉熵 作为损失函数
loss_fn = nn.BCELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
batch_size = 32
steps = X.shape[0] // batch_size

for epoch in range(1000):
    # 每次取32个数据
    for batch in range(steps):
        # 起始索引
        start = batch * batch_size
        # 结束索引
        end = (batch + 1) * batch_size
        # 取数据
        x = X[start:end]
        y = Y[start:end]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # 梯度清零
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 更新操作
        opt.step()
print(model.state_dict())
# 计算正确率
# 设定阈值，现在预测得到的是概率，根据阈值把概率转换为类别，就可以计算准确率
print(((model(X).data.numpy() > 0.5) == Y.numpy()).mean())

# 第二中方式使用继承方式来实现
print('*' * 300)


# pytorch中最常用的创建模型的方式,继承了nn.Module
class Logistic(nn.Module):
    def __init__(self):
        # 先调用父类的方法
        super().__init__()
        # 定义网络中会用到的东西
        self.line_1 = nn.Linear(in_features=23, out_features=64)
        self.line_2 = nn.Linear(in_features=64, out_features=64)
        self.line_3 = nn.Linear(in_features=64, out_features=1)
        self.active = nn.ReLU()  # 激活函数
        self.sigmoid = nn.Sigmoid()  # 非线性变换

    def forward(self, input):
        x = self.line_1(input)
        x = self.active(x)  # 激活
        x = self.line_2(x)
        x = self.active(x)
        x = self.line_3(x)
        x = self.sigmoid(x)
        return x


lr = 0.01


# 定义获取模型的函数和优化器
def get_mode():
    model = Logistic()
    return model, torch.optim.Adam(model.parameters(), lr=lr)


# 定义损失函数
loss_func = nn.BCELoss()
model, optimizer = get_mode()
batch_size = 32
steps = X.shape[0] // batch_size
epochs = 100
for epoch in range(epochs):
    for i in range(steps):
        start = i * batch_size
        end = (i + 1) * batch_size
        x = X[start:end]
        y = Y[start:end]
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('epoch', epoch, 'loss', loss_func(model(X), Y))

print(((model(X).data.numpy() > 0.5) == Y.numpy()).mean())

print('*' * 300)

# 第三种，使用dateset 重构，重构取数据的代码
# start = i * batch_size
# end = (i + 1) * batch_size
# x = X[start:end]
# y = Y[start:end]
# python中的魔术方法：__len__和__getitem__
# len(data) 相当于 data.__len__()
# __getitem__() 相当于根据索引取数据，data[0]=data.__getitem__(0)
# pytorch 中有一个Dataset类，可以把任意具有__len__和__getitem__的对象包装乘Dataset 对象
# Dataset 自动取数据
from torch.utils.data import TensorDataset

dataSet = TensorDataset(X, Y)  # X,Y 都是ndarray
model, optimizer = get_mode()
# 重写训练过程
for epoch in range(epochs):
    for step in range(steps):
        # 取数据不一样
        x, y = dataSet[step * batch_size:(step + 1) * batch_size]
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('epoch', epoch, 'loss', loss_func(model(X), Y))

print(((model(X).data.numpy() > 0.5) == Y.numpy()).mean())

print('*' * 300)
# 第四种，用pytorch 中的dataloader 重构
# dataLoader 可以自动分批取数据，而且dataLoader是由dataset创建出来的
# 由了dataLoader之后就不需要按照切片来取数据
# 一般是先创建一个dataset，然后在创建一个dataloader
from torch.utils.data import DataLoader

ds = TensorDataset(X, Y)
dl = DataLoader(ds, batch_size=batch_size)  # 指定批次
# 现在取数据就可以直接取


model, optimizer = get_mode()
for epoch in range(epochs):
    for x, y in dl:  # dl中的x，y就会按照batchsize 按批次来取数据
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('epoch', epoch, 'loss', loss_func(model(X), Y))

print(((model(X).data.numpy() > 0.5) == Y.numpy()).mean())

print('*' * 300)
## 第五重添加验证
# 需要分割出训练数据和测试数据
# 刚才是把所有数据作为训练数据
from sklearn.model_selection import train_test_split

# 切割数据-》分别创建训练数据和测试数据的dataloader-》训练过程-》校验过程
train_x, test_x, train_y, test_y = train_test_split(X_data.values, Y_data.values.reshape(-1,
                                                                                         1))  # 要传npy的ndarray数据，不能传tensor的数据，切割比例默认是0.25

# train_test_split 是随机切割，默认比例是 0.25
# 想让每次切割一样，可以加随机种子 random_state=5
# 也可以设置train_size 设置切割比例 default = 0.25
# 转化成 tensor
train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
test_y = torch.from_numpy(test_y).type(torch.FloatTensor)
# 再转化乘dataset 和dataLoader
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  #
test_ds = TensorDataset(test_x, test_y)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=True)  # 打乱顺序


# 测试时候的数据可能比较少，可以把batchsize 乘以2

# 定义计算准确率的函数
def accuracy(out, yb):
    return ((out.data.numpy() > 0.5) == yb.numpy()).mean()


# pytorch中有一个训练模式和测试模式 model.trian() 训练模式， model.eval() 测试模式/推理模式
# 训练模式和推理/测试对一些特殊的层会有一些不同的表现，比如dropout,bn 等。
model, optimizer = get_mode()
for epoch in range(epochs):
    # 训练的时候调到训练模式
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每训练10次输出一次测试结果
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 计算测试损失
                valid_loss = sum([loss_func(model(x), y) for x, y in test_dl])  # 平均损失
                acc_mean = np.mean([accuracy(model(x), y) for x, y in test_dl])  # 平均准确率
                print(epoch, valid_loss / len(test_dl), acc_mean)


## 封装各个模块
# 按批次计算损失
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), len(xb)  # 返回的是标量


# 训练

def fit(epochs, model, loss_fn, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_fn, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_fn, xb, yb) for xb, yb in valid_dl]) #计算校验集的损失是不需要传opt
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # 求校验的平均损失
        acc_mean = np.mean([accuracy(model(x), y) for x, y in valid_dl])  # 平均准确率
        print('epoch', epoch, 'loss', val_loss,'accuracy', acc_mean)


def get_data(train_ds, valid_ds, batch_size):
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=True))


# 整个训练校验过程就可以简单用三行代码来完成

train_dl, valid_dl = get_data(train_ds, test_ds, batch_size)
model, opt = get_mode()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
