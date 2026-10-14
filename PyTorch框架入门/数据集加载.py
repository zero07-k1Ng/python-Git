import torch
from torch import nn                          # 神经网络相关工具
from torch.utils.data import Dataset, DataLoader, TensorDataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X          #X：特征数据，通常是形状为 (N, D) 的张量，N 是样本数，D 是特征维度
        self.y = y          #y：标签数据，通常是形状为 (N,) 或 (N, 1) 的张量

    def __len__(self):
        return len(self.X)  #len(self.X) 返回特征张量的第一维大小，即样本数量 N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   # 返回"第 idx 个样本"的特征和标签
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))#生成一个形状为 (1000,) 的一维张量，元素是随机整数。
full_ds = TensorDataset(X, y)
#torch.randint(low, high, size)：生成 [low, high) 区间内的均匀随机整数。
n_train = int(0.8 * len(full_ds))        # 80% 训练，20% 验证
train_ds, val_ds = torch.utils.data.random_split(#随机地将一个数据集切分成多个子集
    full_ds, [n_train, len(full_ds) - n_train]
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)#`batch_size`：每批多少条
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)#`shuffle`：训练集设 True（每轮打乱顺序），验证 / 测试集设 False
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),     #把输入图片调整为指定的尺寸 (64, 64)
    transforms.ToTensor(),           # PIL 图片 → 张量，并自动把像素从 0~255 缩到 0~1
    transforms.Normalize(mean=[0.5], std=[0.5])   #对张量进行标准化（归一化）
])
#把多个变换按顺序组合成一个整体
# train_ds = datasets.ImageFolder("图片根目录", transform=transform)

