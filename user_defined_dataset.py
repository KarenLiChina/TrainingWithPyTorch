import glob

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms

# Dateset: 自定义dataset，要有可迭代的对象，包含 __len__, __getitem__这两个方法，就可以称作是Dataset
# 可以继承Dataset，重写__len__, __getitem__ 这两个方法 __len__ 返回自定义dataset的数量， __getitem__ 根据索引获取里面的数据
# __getItem__ dataset[0]等价于 dataset.__getitem__[0]

# glob.glob('./dataset/') # 可以用相对路径读取
# all_images_path = glob.glob('./dataset/*.jpg')
all_images_path = glob.glob(
    r'D:\code\PythonProjects\AIProjects\TrainingWithPyTorch\dataset\*.jpg')  # 也可以用绝对路径用windos的路径是前面加r，不要转移

# 建立类别和索引直接的关系
species = ['cloudy', 'rain', 'shine', 'sunrise']

species_to_index = dict((c, i) for i, c in enumerate(species))  # enumerate 加了之后就有索引了
# 调换key 和value 直接的顺序
index_to_species = dict((v, k) for k, v in species_to_index.items())  # items是字典，才有key value

# 生成所有图片的label

all_labels = []
for image_path in all_images_path:
    for i, c in enumerate(species):  # c in enumerate(species) 字符串c是否包含在里面 路径
        if c in image_path.split('\\')[-1]:  # 文件名，-1是数组的最后一个
            all_labels.append(i)
# 图片存储的时候相同名字在一起，借助ndarray 的索引取值方法打乱数据
index = np.random.permutation(len(all_images_path))  # 每次执行都是0-length的乱序输出，没有重复

all_images_path = np.array(all_images_path)[index]
all_labels = np.array(all_labels)[index]
# 手动划分训练数据和测试数据

split = int(len(all_images_path) * 0.8)
train_imgs = all_images_path[:split]
train_labels = all_labels[:split]

test_imgs = all_images_path[split:]
test_labels = all_labels[split:]

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), # 把非 RGB的转换成RGB
    transforms.ToTensor()
])  # 一个简单的transform，没有区分train和test


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        # 根据index 获取item
        img_pth = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img_pth)
        data = self.transforms(img)
        return data, label

    def __len__(self):
        return len(self.imgs)

train_dataset = MyDataset(train_imgs, train_labels, transform)
test_dataset = MyDataset(test_imgs, test_labels, transform)

batch_size = 16
train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True) # drop掉最后一个不满足尺寸的数据集
test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*2,drop_last=True)# 测试数据集不用打乱顺序，size 可以大一点


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


best_acc = 0
epochs = 20
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for epoch in range(epochs):
    epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = fit(epochs=epoch, model=model, train_dl=train_loader,
                                                                 test_dl=test_loader)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(test_epoch_loss)
    test_acc.append(test_epoch_acc)
    if test_epoch_acc > best_acc:  # 关注测试准确率
        best_acc = test_epoch_acc


