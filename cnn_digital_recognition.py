import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  # 如果GPU可用，就用GPU，cuda 0 第一个GPU，独立显卡，否则用CPU
# 在pytorch 中使用GPU进行训练
# 1. 把模型转移到GPU上，需要用代码显示用GPU上运行
# 2. 把每一批次的训练数据转移到GPU上
# torchvision内置了常用的数据集合和常见的模型
# transforms 在pytorch中用来做数据增强和数据预处理等功能
transformation = transforms.Compose([transforms.ToTensor()])  # 把数据预处理的方法的组合
# transforms.ToTensor() 的作用有三个：
# 1. 把数据转换为tensor，
# 2. 数据的值转换到0-1之间，
# 3. 会把通道channel 放到第一个维度上，一般一个彩色的图是28*28*3， 通道是在后面，但是pytorch放到最前面，就是3*28*28，对于一个黑白图片就是1*28*28
train = datasets.MNIST(root='./', train=True, transform=transformation, download=True)  # 会自动下载
# 测试数据集
test_ds = datasets.MNIST(root='./', train=False, transform=transformation, download=True)
# 转换乘dataLoader

train_dl = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)  # 测试数据可以给大一点，不需要反向传播，也不需要打乱
images, labels = next(iter(train_dl))  # 通过iter变成迭代器,通过next取出一组数据,一批是64个
print(images.shape)  # pytorch 中的图片的表现形式[batch, channel, hight, weight] [64,1,28,28] 一批是64个，1个通道，28*28的数据
img = images[0]
img = img.numpy()  # 黑白图，给它降维
img = np.squeeze(img)  # 降维，把1维的通道去掉


# plt.imshow(img, cmap='gray')
# plt.show() 展示第一个图片
## 创建模型
class DigitalRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # 神经网络中等卷积 Conv2d 的参数
        # in_channels 输入通道，灰度图片维1， RGB彩色图片为3
        # out_channels 输出通道数,就是卷积核的个数，输出特征图的通道数，决定这一层学习多少个不同特征
        # kernel_size，卷积核的大小 kernel_size=3是3*3的卷积核，kernel_size=(3,5) 是3*5 矩形卷积核
        # stride，步长，默认是1，每次移动几个像素
        # padding，输入四周的填充层数，默认是0 不填充
        # dilation，是否膨胀，默认为1，不膨胀
        # group 是否分组
        # bias 是否使用偏置项 bias=True
        # padding_mode padding_mode='zeros' padding模式是否补0
        self.pool = nn.MaxPool2d(2)  # 池化 kernel_size 设置为2，池化可以反复用
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # 定义一层全连接
        # 第一次卷积in [64,1,28,28]->out [64,32,26,26]-> 池化 64，32，13，13
        # ->第二次卷积 out[64,64,11,11]->再加一层池化 out 64，64，5，5
        self.linear_1 = nn.Linear(in_features=64 * 5 * 5, out_features=256)
        self.linear_2 = nn.Linear(in_features=256, out_features=10)  # 输出层

    def forward(self, input):  # 前向传播
        x = F.relu(self.conv1(input))  # 用relu激活函数
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # flatten,做一个变形
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear_1(x))
        self.linear_2(x)
        return x


model = DigitalRecognizer()
# 把model拷到GPU上
model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def fit(epochs, model, train_dl, valid_dl):
    correct = 0
    total = 0
    running_loss = 0
    for x, y in train_dl:
        # 需要把x，y放到GPU上
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)  # 取第二个维度的，就是真实的预测结果
            correct += (y_pred == y).sum().item()
            total += y.size(0)  # 样本个数
            running_loss += loss.item()
    epoch_loss = running_loss / len(train_dl.dataset)
    epoch_acc = correct / total

    # 测试过程
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)  # 取第二个维度的，就是真实的预测结果
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)  # 样本个数
            test_running_loss += loss.item()
    test_epoch_loss = test_running_loss / len(test_dl.dataset)
    test_epoch_acc = test_correct / test_total

    print('epoch:', epochs,
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test loss: ', round(test_epoch_loss, 3),
          'test accuracy:', round(test_epoch_acc, 3))
    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc


epochs = 20
train_loss=[]
train_acc=[]
valid_loss=[]
valid_acc=[]
for epoch in range(epochs):
    epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = fit(epochs=epoch, model=model, train_dl=train_dl,valid_dl=test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    valid_loss.append(test_epoch_loss)
    valid_acc.append(test_epoch_acc)