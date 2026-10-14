import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):          # 假设 5 种病害
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),#卷积层。输入3个通道（RGB），输出16个特征图。卷积核大小3x3，四周补一圈0（padding=1）
            nn.BatchNorm2d(16),#批归一化。把16个通道的数据分布拉回标准正态
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 最大池化，用 2x2 的窗口，步长为2

            nn.Conv2d(16, 32, 3, padding=1),              # (N,32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (N,32,16,16)

            nn.Conv2d(32, 64, 3, padding=1),              # (N,64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (N,64,8,8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),#把 (N, 64, 8, 8) 拍扁成 (N, 4096)。4096 正好是 64 * 8 * 8
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),     # 输出 5 个类别的 logits
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN(num_classes=5)
x = torch.randn(32, 3, 64, 64)
print(model(x).shape)          # torch.Size([32, 5])

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),        # 数据增强：随机翻转，仅训练用，缓解过拟合
    transforms.ToTensor(),                    # 图片→张量，像素 0~255 自动缩到 0~1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 用 ImageNet 统计量标准化
                         std =[0.229, 0.224, 0.225]),  #分别对应红(R)、绿(G)、蓝(B)三个颜色通道
])
train_ds = datasets.ImageFolder("data/train", transform=train_tf)#"data/train"：这是图片所在的根目录路径
#transform=train_tf：这是你上面定义好的那个“流水线”。这意味着，当 ImageFolder 从硬盘里读出一张图片时，它会立刻顺手把 train_tf 里的所有操作（Resize → 随机翻转 → 转张量 → 标准化）全部应用上去。
#最终得到的 train_ds 里面的每一张图，都是已经处理好的张量
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)#下载并加载在 ImageNet（1000万+张图片）上已经训练好的最佳权重
model.fc = nn.Linear(model.fc.in_features, 5)   # 把原 1000 类输出头换成 5 类病害
# 之后用第 1 部分的训练循环正常训练即可（lr 可设小一点，如 1e-4）
