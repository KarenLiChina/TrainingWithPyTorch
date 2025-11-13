import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms

# 数据增强， 放到transforms.Compose 中,解决过拟合、数据量不足，提高模型返回能力
# 基于 cnn_weather_recognition.py 中的数据路径
# 从https://www.kaggle.com/datasets/saurabhshahane/multi-class-weather-dataset 下载数据，放到项目的./dataset 中

base_dir = './dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 数据增强的一些方式
# transforms.Random* 随机的一些转换，裁剪，锐化，反转，旋转
# transforms.CenterCrop 从中间剪裁
# transforms.RandomRotation 随机旋转
# transforms.RandomHorizontalFlip 随机水平反转
# transforms.RandomVerticalFlip 随机垂直反转
# transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1) 对某个数据进行抖动 brightness 亮度，contrast 对比度，saturation饱和度，hue 颜色
# transforms.RandomGrayscale 随机灰度化
# transforms.RandomCrop 随机裁剪

# 数据增强只会加在训练数据上
train_transfrom = transforms.Compose([
    transforms.Resize((224, 224)),  # 专门进行图片缩放的方法，都转换为224*224
    transforms.RandomCrop(192), #随机剪切图片
    transforms.RandomHorizontalFlip(), # 水平反转 默认50%的可能性被反转
    transforms.RandomVerticalFlip(), # 垂直反转
    transforms.RandomRotation(0.4),# 随机选择的角度，给一个数字，就按照这个数字的-degree，degree去选择，给两个数字，就在这个范围内旋转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # 正则化，mean 均值，std标准差，这些数值是在ImageNet数据集上计算得到的统计量,ImageNet: 包含1400万张图像，22000个类别,在数百万张图像上计算每个通道的均值和标准差
])
# 测试数据不需要数据增强
test_transfrom = transforms.Compose([
    transforms.Resize((224, 224)),  # 专门进行图片缩放的方法，都转换为224*224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # 正则化，mean 均值，std标准差，这些数值是在ImageNet数据集上计算得到的统计量,ImageNet: 包含1400万张图像，22000个类别,在数百万张图像上计算每个通道的均值和标准差
])
train_ds = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transfrom)  # 训练数据集，要求同类的数据在同一个文件夹内
test_ds = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transfrom)
batch_size = 32
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       drop_last=True)  # 最后一批数据如果不满足数量就丢掉
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size * 4)  # 测试数据不需要随机

model = torchvision.models.vgg16(
    pretrained=True)  # BN 是2015年出来的，会有带bn和不带bn版本的模型，pretrained 为True加载在 ImageNet 数据集上预训练好的权重,第一次加载会下载 模型
for param in model.features.parameters():
    param.requires_grad = False  # 原有模型中的参数，设置为不可训练，保持和之前一样

# 修改原网络输出层结构
# model.classifier[-1].out_features = 4 #原模型最后的输出类别为1000，即可以识别1000种类，我们要修改为4，输出类别为4种
# 另一种修改输出层的写法，把输出层直接改掉
model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=4)

# 拷贝到 GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  # 如果GPU可用，就用GPU，cuda 0 第一个GPU，独立显卡，否则用CPU

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 学习率衰减 pytorch 里封装好的 stepLR 根据训练次数来衰减，学习率越到后面就越小，step_size多少步衰减一次，gamma 默认是0.1 衰减率
step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def fit(epochs, model, train_dl, test_dl):
    correct = 0
    total = 0
    running_loss = 0
    model.train()  # 模型训练
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
    step_lr_scheduler.step()
    epoch_loss = running_loss / len(train_dl.dataset)
    epoch_acc = correct / total

    # 测试过程,测试过程是推测，不需要学习率衰减
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()  # 模型推理
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


epochs = 10  # 训练10次就可以了，vgg比较慢
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for epoch in range(epochs):
    epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = fit(epochs=epoch, model=model, train_dl=train_dl,
                                                                 test_dl=test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(test_epoch_loss)
    test_acc.append(test_epoch_acc)
# 第一次训练表现就很好

plt.plot(range(1, epochs + 1), train_loss, label='train loss')
plt.plot(range(1, epochs + 1), test_loss, label='test loss')
plt.legend()
plt.show()

plt.plot(range(1, epochs + 1), train_acc, label='train accuracy')
plt.plot(range(1, epochs + 1), test_acc, label='test accuracy')
plt.legend()
plt.show()
