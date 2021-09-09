# 学习笔记

### 目录
1. [Dataset](#Dataset)
2. [TensorBoard](#TensorBoard)
3. [Transforms](#Transforms)
4. [torchvision内置数据集](#torchvision)
5. [DataLoader](#DataLoader)
6. [nn.Module-conv](#nn.Module)
7. [Pooling](#Pooling)

## Dataset
<a id="Dataset"></a>

|   name   |          work            |
| -------- | ------------------------ |
|  Dataset |提供一种方式去获取数据及其label|
|Dataloader|为后面的网络提供不同的数据形式  |

#### 引入Dataset
```python
from torch.utils.data import Dataset
```

#### 创建类继承Dataset
````python
class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)
````
#### 路径设置
1. 绝对路径:
   
   `r"C:\Users\Lty\Desktop\PytorchTest\image\pytorch.jpeg"`或
   `"C:\\Users\\Lty\\Desktop\\PytorchTest\\image\\pytorch.jpeg"`
2. 相对路径:  
   
   `"image/pytorch.jpeg"`

#### os模块常用方法
```python
import os
path="./"

os.chdir(path) #改变当前工作目录
os.listdir(path) #列举指定目录的文件名
os.mkdir(path) #创建path指定的文件夹,只能创建一个单层文件，而不能嵌套创建
os.makedirs(path) #创建多层目录 ，可以嵌套创建

#os.path模块
os.path.abspath(path) #返回文件或目录的绝对路径
os.path.dirname(path) #返回path路径最后一个\\之前的内容
os.path.split(path) #返回一个（head,tail）元组，head为最后\\之前的内容；tail为最后\\之后的内容，可以为空
os.path.join(path,*path) #将两个path通过\\组合在一起，或将更多path组合在一起

```

## TensorBoard
<a id="TensorBoard"></a>

#### 简介
TensorBoard是一个可视化工具，它可以用来展示网络图、张量的指标变化、张量的分布情况等。特别是在训练网络的时候，我们可以设置不同的参数（比如：权重W、偏置B、卷积层数、全连接层数等），使用TensorBoader可以很直观的帮我们进行参数的选择。

#### 使用方法
````python
# 1.引入tensorboard
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# 2.创建实例
writer = SummaryWriter("logs")

# 3.显示数据
# y=2x
for i in range(100):
    writer.add_scalar("y=x",2*i,i)
# 4.显示图像
image_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# 4.关闭SummaryWriter
writer.close()
````
#### 命令行运行
`tensorboard --logdir=logs --port=6007`

#### add_image方法
```python
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
```
#### 效果
![](https://s3.bmp.ovh/imgs/2021/09/3983e3462e531bb8.png)
![](https://s3.bmp.ovh/imgs/2021/09/0caa7c0dd0fe4bbe.png)

## Transforms
<a id="Transforms"></a>
#### 作用
torchvision.transforms : 常用的图像预处理方法，提高泛化能力
0. 图像类型转化
1. 数据中心化
2. 数据标准化
3. 缩放、裁剪、旋转、翻转、填充
4. 噪声添加
5. 灰度变换
6. 线性变换
7. 亮度、饱和度及对比度变换

#### 运行机制
采用transforms.Compose()，将一系列的transforms有序组合，实现时按照这些方法依次对图像操作。
```python
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 缩放
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
    transforms.Normalize(norm_mean, norm_std),  # 标准化均值为0标准差为1
])
```
#### 常用图片类型
type | function | transforms
---- | -------- | ----------
 PIL image| Image.open | ToPILImage
Tensor | ToTensor | ToTensor/PILToTensor
narrays | cv.imread | ConvertImageDtype(任意类型)

(为何深度学习需要使用tensor类型：主要原因，tensor类型具有反向传播属性)

#### 常用方法
1. ToTensor
```python
# ToTensor
img = Image.open("image/pytorch.jpeg")
#PIL image -> Tensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
```
2. Normalize(归一化)
```python
# output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize([1, 3, 5], [2, 4, 6])
img_norm = trans_norm(img_tensor)
```
3. Resize
```python
trans_resize = transforms.Resize((512, 512))
# img PIL ->resize ->img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> toTensor -> img_resize tensor
img_resize = trans_toTensor(img_resize)
```

|  name   | function |
|  ----  | ----  |
| 修改亮度、对比度和饱和度  | transforms.ColorJitter |
| 转灰度图  | transforms.Grayscale |
| 线性变换  | transforms.LinearTransformation |
| 中心裁剪  | transforms.CenterCrop|
|上下左右中心裁剪|transforms.FiveCrop|
| 依概率p水平翻转 | transforms.RandomHorizontalFlip(p=0.5)|
| 依概率p垂直翻转 | transforms.RandomVerticalFlip(p=0.5) |

## torchvision内置数据集
<a id="torchvision"></a>

#### 常用内置数据集
![](https://s3.bmp.ovh/imgs/2021/09/1db657050d1a91d6.png)

#### 数据集加载
```python
# 参数配置，以Cifar10为例
# Args:
#         root (string): Root directory of dataset where directory
#             ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
#         train (bool, optional): If True, creates dataset from training set, otherwise
#             creates from test set.
#         transform (callable, optional): A function/transform that takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./vision_dataset", train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root="./vision_dataset", train=False, download=False)
```

## DataLoader
<a id="DataLoader"></a>
#### 常用配置参数
|  name   | work |
|  ----  | ----  |
| dataset  | 所用数据集 |
| batch_size  | 每批次读取的数量 |
| shuffle  | 是否打乱数据 |
| drop_last | 是否舍弃未满一批次的数据|
| num_workers| 读取开始位置 |

#### 使用示例
```python
import torchvision
from torch.utils.data import DataLoader
# 准备测试集
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./vision_dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
# print(img.shape)
# print(target)
writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step+1

writer.close()
```

#### tensorboard工具下可视化结果
```
tensorboard --logdir="dataloader"
```

## nn.Module-conv
<a id="nn.Module"></a>

#### torch.nn
[![hbvtk8.png](https://z3.ax1x.com/2021/09/09/hbvtk8.png)](https://imgtu.com/i/hbvtk8)
#### torch.nn.Module
```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

#### 卷积操作
![](https://www.hualigs.cn/image/6139708b914fc.jpg)

![](https://www.hualigs.cn/image/613970a6e8c13.jpg)

| Parameters | mean |
| --- | --- |
| in_channels (int) | 输入个数 |
| out_channels (int) | 输出个数 |
| weight | 卷积核维数 |
| bias | 偏置 |
| stride | 步径 |
| padding | 对输入进行填充 |

#### 原理
![](https://www.hualigs.cn/image/613970ea9c491.jpg)
#### 代码实现
```python
import torch
import torch.nn.functional as F

input = torch.tensor([[1,0,1],
                      [0,2,1],
                      [1,2,0]])
kernel = torch.tensor([[1,0],
                      [0,2]])
input = torch.reshape(input,(1,1,3,3))
kernel =torch.reshape(kernel,(1,1,2,2))
output = F.conv2d(input,kernel,stride=1)
print(output)
# tensor([[[[5, 2],
#           [4, 2]]]])
```
#### 可视化示例

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="https://s3.bmp.ovh/imgs/2021/09/e5a050575544c640.gif"></td>
    <td><img width="150px" src="https://s3.bmp.ovh/imgs/2021/09/ffbf1462c3215081.gif"></td>
    <td><img width="150px" src="https://s3.bmp.ovh/imgs/2021/09/ec1fe4a0ad20f685.gif"></td>
    <td><img width="150px" src="https://s3.bmp.ovh/imgs/2021/09/6dc2712ff2bb1726.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
    <td>Full padding, no strides</td>
  </tr>
  <tr>
    <td><img width="150px" src="https://s3.bmp.ovh/imgs/2021/09/ddffae6dbf069245.gif"></td>
    <td><img width="150px" src="https://s3.bmp.ovh/imgs/2021/09/aa8363df84c53b90.gif"></td>
    <td><img width="150px" src="https://s3.bmp.ovh/imgs/2021/09/c506ea8e90be7bdd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
    <td></td>
  </tr>
</table>

## Pooling
<a id="Pooling"></a>

![](https://www.hualigs.cn/image/6139712c20c5c.jpg)

#### 最大池化原理
![](https://www.hualigs.cn/image/6139712c838d6.jpg)

#### 参数（与卷积层类似）
![](https://www.hualigs.cn/image/6139712cb38b6.jpg)

#### 代码实现
```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1,0,1],
                      [0,2,1],
                      [1,2,0]],dtype=torch.float32)
input = torch.reshape(input,(1,1,3,3))
print(input.shape)
# torch.Size([1, 1, 3, 3])

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=2, ceil_mode=True,stride=1)

    def forward(self,input):
        output = self.maxpool1(input)
        return output


model1 = MyModel()
output = model1(input)
print(output)
# tensor([[[[2., 2.],
#           [2., 2.]]]])
```
