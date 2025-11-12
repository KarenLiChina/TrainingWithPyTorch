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

