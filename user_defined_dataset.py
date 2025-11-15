import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
