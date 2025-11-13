## python环境要求
```bash
pip install -r requirements.txt
```

## 安装pytorch
### CPU
```bash
pip install torch torchvision

```

### GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## 查看 GPU

ipython
import torch
torch.cuda.is_available() 
torch.cuda.get_device_name(0)

## cnn_weather_recognition 中的图片数据来源 kaggle
https://www.kaggle.com/datasets/saurabhshahane/multi-class-weather-dataset

### 构建卷积神经网络，对MNIST中的灰度数字进行识别，将模型放到GPU上进行训练
cnn_digital_recognition.py
### 对kaggle中四种天气进行卷积神经网络分类，增加dropout（随机关闭一些神经节）和BN层（批量归一化）
cnn_weather_recognition.py
### 基于现有模型vgg进行迁移学习
transfer_learning_base_vgg.py
### 学习率衰减，越接近参数值时，学习率应该变得小一点，越接近于极值
decay_rate.py
### 数据增强， 放到transforms.Compose 中,解决过拟合、数据量不足，提高模型返回能力
data_augmentation.py 