# pytorch 中的张量和tensorflow中的tensor是一样的，名字都一样
# pytorch 中的张量也叫 tensor，就是多维数组
# tensor和numpy 中的ndarray 也是一个意思，只不过tensor可以在GPU上加速计算
import numpy as np
import torch
from sympy import print_glsl

# 创建tensor
torch.tensor([6, 2], dtype=torch.int32)  # 用一维数组初始化tensor，类型必须是 torch里的类型
torch.tensor((6, 2))  # 也可以用元组初始化tensor
torch.tensor(np.array([6, 2]))  # 用np.array 也可以

# 快速创建tensor的方法，和numpy 中的routines方法一样
# ones, zeros, full, eye, random.randn, random.normal... arange...
# 创建一个0-1之间的随机数组成的tensor
torch.rand((2, 3))  # 创建两行，三列0-1的随机数
torch.rand((2, 3))  # 创建两行，三列0-1的随机数,，两种写法都可以

# 标准正态分布
torch.randn(2, 3)  # 创建两行，三列，标准整体分别的随机数

# torch.normal() #需要传参数 均值，标准偏差，生成器，从生成器中生成

torch.zeros(2, 3)  # 创建两行，三列的0
torch.ones(2, 3)  # 创建两行，三列的1

# tensor中的属性
# tensor的shape

x = torch.ones(2, 3, 4)  # 三维
print(x.shape)  # tensor的shape
print(x.size())  # 也可以取到tensor的形状
# size中的索引，也是从0开始的
x.size(0)  # shape的索引，0-》2，1-》3，2-》4

# tensor中等 基础数据类型，位数越长，可以保存数据的精度越精准
print(torch.float32)  # 32位浮点数
print(torch.float64)  # 64位浮点数
print(torch.int32)  # 32位整数
print(torch.int16)  # 16位整数
print(torch.int64)  # 64位浮点数整数

# 可以在创建tensor时候指定数据类型，必须用torch中自己的类型
torch.tensor([1, 2, 3], dtype=torch.int32)
# import tensorflow as tf

# tensorflow 没有直接用tenser创建tenser的方法,下面的方法不可用
# tf.tensor([6,2])
# tf.Tensor([1, 2, 3])，有这个方法，不是用来创建tensor
# tensorflow 提供了constant，variable，必须是常量或变量
# pytorch 中不区分是常量还是变量，都是变量

# tensor 可以和ndarray 很方便的进行转换
n = np.random.randn(2, 3)  # ndarray
a = torch.from_numpy(n)  # 从ndarray to tensor
a.numpy()  # 转换位ndarray

# t = tf.constant(a)  # tensorflow 可以从 tensor 中转换

# 张量运算
# tensor 运算规则 和numpy的ndarray很想
# 和单个数字运算

t = torch.ones(2, 3)
print(t + 3)  # 每个变量都加3
print(torch.add(t, 3))  # 和上面的结果一样，可以用上面的用法，直接加
# 两个tensor相加时，要么是shape 形状相同（行列相同），要么是用广播的形式
# 广播 必须满足两个条件
# 1. 从尾部维度开始对齐（从右向左）
# 2. 每个对应的维度必须满足以下条件之一：维度大小相等、其中一个维度大小为1、其中一个张量在该维度上不存在维度
x1 = torch.ones(2, 3)
print(x1 + t)  # 对应位置元素相加，element-wise操作，就是残差网络的加法，就是这种加法，就是element-wise相加，要求形状相同
t.add(x1)  # 也可以用这种方式，不会改变原来的值
t.add_(x1)  # pytorch 加下划线 _ 操作会改变原始值

# 求逆，必须给一个矩阵 torch.inverse()
t.reshape(3, 2)  # 改变形状为三行，两列
t.view(3, 2)  # 改变形状为三行，两列

# 聚合操作
t.mean()  # 均值
t.sum()  # 求和，默认所有维度聚合
x = t.sum(dim=1)  # 指定维度聚合
print(x)
# 一个数字是标量 scalars,带括号的数据叫做向量，item是专门用来取出tensor中的标量的值
# x.item() 此时会报错
x = x.sum()
x.item()  # 此时为标量，不报错
x = torch.randn(3, 4)
print(x[0])  # 第0行
print(x[0, 0])  # 第0行，0列
print(x[0, :3])  # 切片也可以，和ndarray是完全一样的
x = torch.rand(32, 224, 224, 3)  # 卷积神经网络的写法 32是批次大小，224*2224 厮图片的长宽，3是RGB
# 取出图片数据
x[0, :, :, 0]  # 取出来的是个二维数组，shape是224，224

# 矩阵乘法，第一个矩阵队和第二矩阵的行相同才可以乘
a1 = torch.randn(2, 3)
a2 = torch.randn(3, 5)
torch.matmul(a1, a2)  # 结果为2行5列的矩阵
print(a1 @ a2)  # @为矩阵点乘的快捷写法

# pytorch中的dot点乘，是向量的内积，两个向量各个元素相对应的位置相乘后相加
x1 = torch.randn(5)
x2 = torch.randn(5)
print(x1)
print(x2)
print(x1.dot(x2))

# 张量的自动微分
# 就是自动求导
x = torch.ones(2, 2, requires_grad=True)  # 在要求导的变量定义中加上requires_grad=True，表示要追踪这个变量的导数
y = x + 2
print(x.requires_grad)  # 可以通过这个语句查到是否可以求导
x2 = torch.ones(2, 2)  # requires_grad 默认值是False，默认是不求导的
print(x2.requires_grad)  # False

print(y.grad_fn)  # backward，反向传播，grad_fn记录了x对y反向传播的过程

# 要用backward来求导,导数必须由标量输出创建，要求的最后结果必须是一个标量，不能是一个向量
# print(y.backward()) # 会遇到后面的错误：grad can be implicitly created only for scalar outputs

z = y.mean()  # 平均值是一个值，就是标量，可以进行反向传播，求导
z.backward()  # 执行backward之后，所有requires_grad=True的变量都会被自动求出，如果不执行这步，后面导数就没有了，一定要执行
print(x.grad)  # 可以求出了x的导数值

x = torch.ones(2, 2, requires_grad=True)  # 在要求导的变量定义中加上requires_grad=True，表示要追踪这个变量的导数
y = x + 2
z = y * y * 3  # 符合函数
out = z.mean()
y.retain_grad()  # 重要：在 backward() 之前调用！ x->y->z->out,求非叶子节点的tensor的导数，必须用retain_grad，不然会报警，requires_grad是True，y是由x算出来的
out.backward()  # 然后进行反向传播

print(y.grad)
print(x.grad)

# 如果不需要求导，可以把代码包在torch.no_grad()中，相当于临时不求导
print(x.requires_grad)
print((x ** 2).requires_grad)  # 也是可以求导

# 通过with torch.no_grad() 暂时不对x求导
with torch.no_grad():
    print((x ** 2).requires_grad)  # 在这段上下文中是不求导的

y = x ** 2 + 2  # 此时y也是可以求导的
y = x.detach()  # 脱离接触，也会不可求导
print(y.requires_grad)

# 除了在定义的时候指定requires_grad,也可以通过方法 requires_grad_()方法修改requires_grad的属性
a = torch.randn(2, 3)
a = a * 3 + 2
print(a.requires_grad)  # 不可求导
a.requires_grad_(True)  # 通过这种方式设置是否可以求导，加下划线会修改原始数据
print(a.requires_grad)  # 可以求导
