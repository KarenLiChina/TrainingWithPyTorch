import copy
import os
import shutil

import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import transforms

# 从https://www.kaggle.com/datasets/saurabhshahane/multi-class-weather-dataset 下载数据，放到项目的./dataset 中
base_dir = './dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

images_names = os.listdir('./dataset')
species = ['cloudy', 'rain', 'shine', 'sunrise']
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

for specie in species:
    if not os.path.exists(os.path.join(train_dir, specie)):
        os.makedirs(os.path.join(train_dir, specie))
    if not os.path.exists(os.path.join(test_dir, specie)):
        os.makedirs(os.path.join(test_dir, specie))

# enumerate 给迭代对象添加索引
for i, image_name in enumerate(images_names):
    for specie in species:
        if specie in image_name and image_name.endswith('.jpg'):
            image_path = os.path.join(base_dir, image_name)
            if i % 5 == 0:
                target_path = os.path.join(test_dir, specie, image_name)
            else:
                target_path = os.path.join(train_dir, specie, image_name)
            if not os.path.exists(target_path):
                shutil.copy(image_path, target_path)  # python中自带一个好用的拷贝工具
for train_or_test in ['train', 'test']:
    for specie in species:
        # 打印各类数据数量
        print(train_or_test, specie, len(os.listdir(os.path.join(base_dir, train_or_test, specie))))

transfrom = transforms.Compose([
    transforms.Resize((96, 96)),  # 专门进行图片缩放的方法，都转换为96*96
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # 正则化，mean 均值，std标准差，这些数值是在ImageNet数据集上计算得到的统计量,ImageNet: 包含1400万张图像，22000个类别,在数百万张图像上计算每个通道的均值和标准差
])
train_ds = torchvision.datasets.ImageFolder(root=train_dir, transform=transfrom)  # 训练数据集，要求同类的数据在同一个文件夹内
test_ds = torchvision.datasets.ImageFolder(root=test_dir, transform=transfrom)

print(train_ds.classes)  # 目录下有多少个文件夹，就有多少个类别
print(train_ds.class_to_idx)  # 类别对应的索引
print(len(train_ds))
print(len(test_ds))
batch_size = 32
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size * 4)  # 测试数据不需要随机

images, labels = next(iter(train_dl))  # 取到的是一批图片和对应的标签
print(labels)  # label已经做过映射，和的之前train_ds.class_to_idx 中的索引一致


# squeeze 只能把维度是1的地方压缩掉，（28，28，1）-》（28，28）
# reshape 也不可以，会打乱数据

# torch.transpose(images, 0, 1)
# 输出展示下图片，之前做过调整，image的数据不是在0-1 范围内，需要缩放
# img = images[0]
# print(img.max())
# print(img.min())
# img = img + 2.2
# img = img / 4.7
# print(img.max())
# print(img.min())
# plt.imshow(img.permute(1, 2, 0))  # 重新把维度进行 排列组合
# plt.show()

# 定义模型

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  # 输入三个通道RGB'，16个卷积核，卷积核3*3,步长没写默认为1, 16*94*94
        self.pool = nn.MaxPool2d(2, 2)  # 卷积核，步长 16*94/2*94/2 =16*47*47
        self.conv2 = nn.Conv2d(16, 32, 3)  # 32 *45*45 ->再经一轮池化 pooling 32 * 22*22
        self.conv3 = nn.Conv2d(32, 64, 3)  # 64 * 20 *20 ->再经一轮池化 pooling 64 * 10*10
        self.dropout = nn.Dropout(
            p=0.5)  # p默认值也是0.5,防止过拟合，每个神经元都有一定的概率 p（例如0.5）被设置为零（被“关闭”），同时剩下的神经元（概率为 1-p）其值会被放大 1/(1-p) 倍
        # batch, channel, height, width, 64, 3,
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)  # 第一个参数为输入，
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积-》激活-》池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积-》激活-》池化
        x = self.pool(F.relu(self.conv3(x)))  # 卷积-》激活-》池化
        x = nn.Flatten()(x)  # 等价于 x.view(-1,64*10*10)，做一次变形
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class NetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  # 输入三个通道RGB'，16个卷积核，卷积核3*3,步长没写默认为1, 16*94*94
        self.bn1 = nn.BatchNorm2d(16)  # bn 批量归一化，需要单独写，不能像pool共用,BN 有两个要学习的参数，进行整体偏移，训练模式时，需要反向求导，推导模式需要固定
        # 2d是只每一个通道（RGB）有长宽两个维度，BN 一般放到卷积之后，上一输出维16
        self.pool = nn.MaxPool2d(2, 2)  # 卷积核，步长 16*94/2*94/2 =16*47*47
        self.conv2 = nn.Conv2d(16, 32, 3)  # 32 *45*45 ->再经一轮池化 pooling 32 * 22*22
        self.bn2 = nn.BatchNorm2d(32)  # 上一层输出为32
        self.conv3 = nn.Conv2d(32, 64, 3)  # 64 * 20 *20 ->再经一轮池化 pooling 64 * 10*10
        self.bn3 = nn.BatchNorm2d(64)  # 上一层输出为64
        self.dropout = nn.Dropout(
            p=0.5)  # p默认值也是0.5,防止过拟合，每个神经元都有一定的概率 p（例如0.5）被设置为零（被“关闭”），同时剩下的神经元（概率为 1-p）其值会被放大 1/(1-p) 倍
        # batch, channel, height, width, 64, 3,
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)  # 第一个参数为输入，
        self.bn_fc1 = nn.BatchNorm1d(1024)  # 上一层输出是1维，输出为1024
        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)  # 上一次输出为256
        self.fc3 = nn.Linear(256, 4)  # 最后一层输出，不需要bn

    def forward(self, x):
        # 正确的顺序：卷积 -> BN -> 激活 -> 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = nn.Flatten()(x)  # 等价于 x.view(-1,64*10*10)，做一次变形
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = NetWithBN()
# 把模型拷到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  # 如果GPU可用，就用GPU，cuda 0 第一个GPU，独立显卡，否则用CPU

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


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
    epoch_loss = running_loss / len(train_dl.dataset)
    epoch_acc = correct / total

    # 测试过程
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


best_model_weight = model.state_dict()
best_acc = 0
epochs = 20
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
    if test_epoch_acc > best_acc:  # 关注测试准确率
        best_acc = test_epoch_acc
        best_model_weight = copy.deepcopy(model.state_dict())  # 更新参数，需要深拷贝，不能用赋值的浅拷贝

# 保存参数
weight_path = './model/weather_recognition_weight.pth'  # 将参数保存到文件中，是一个序列化的结构，保存的结构比较大
torch.save(best_model_weight, weight_path)  # 最好的模型保存下来

# 从存储的模型中
new_model = NetWithBN()  # 创建一个新的模型
# new_model.load_state_dict(best_model_weight)  # 直接通过模型参数load模型
new_model.load_state_dict(torch.load(weight_path))  # 通过torch.load(patch)加载了我们之前存储的参数的文件，得到了之前存储的模型

# 保存完整模型
model_path = './model/weather_recognition_model.pth'  # 保存完整模型
torch.save(model, model_path)
new_model_2 = torch.load(model_path, weights_only=False)  # 通过torch.load(path) 来load 整个模型
# pytorch 2.6 之后 weights_only 的默认值从false 改为True了，要显示指定weights_only 为False 来信任模型来源

# 用新的模型放到GPU上，进行测试
new_model.to(device)
test_correct = 0
test_total = 0
new_model.eval()
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        y_pred = new_model(x)
        y_pred = torch.argmax(y_pred, dim=1)  # 取第二个维度的，就是真实的预测结果
        test_correct += (y_pred == y).sum().item()
        test_total += y.size(0)  # 样本个数
epoch_test_correct = test_correct / test_total
print("new mode test correct: ", epoch_test_correct)

# 跨设备的模型保存和加载，可以在CPU/GPU直接任意保存和加载
# 把刚才保存的模型映射到GPU上去
new_model_3 = NetWithBN()
# 相当于执行完加载模型后，又执行了model.to(device)
new_model_3.load_state_dict(torch.load(weight_path,map_location=device))
# 图表展示
plt.plot(range(1, epochs + 1), train_loss, label='train loss')
plt.plot(range(1, epochs + 1), test_loss, label='test loss')
plt.legend()
# plt.show()

plt.plot(range(1, epochs + 1), train_acc, label='train accuracy')
plt.plot(range(1, epochs + 1), test_acc, label='test accuracy')
plt.legend()
# plt.show()
