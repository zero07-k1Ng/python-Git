import torch
from torch import nn                          # 神经网络相关工具
from torch.utils.data import Dataset, DataLoader, TensorDataset
'''
nn.Linear(in_features, out_features)：全连接层，做 `输出 = 输入 @ 权重 + 偏置`（一次矩阵乘法加偏置）
激活函数：`nn.ReLU()`（把负数变 0、正数保留，引入非线性，让网络能学复杂关系）、`nn.Sigmoid()`、`nn.Softmax()`
`nn.Flatten()`：把图片等多维输入拉平成向量
`nn.Sequential(...)`：把若干层按顺序**打包成一条流水线**
'''
model = nn.Sequential(
    nn.Linear(10, 16),   # 输入10个特征,输出维度：16
    nn.ReLU(),           # 非线性激活
    nn.Linear(16, 2),    # 16 → 输出2个类别的分数
)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()              #super() 指的是父类 nn.Module,调用父类的初始化方法,固定写法
        self.fc1 = nn.Linear(10, 16)    # 定义零件并挂到 self 上
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)                 # 数据流过第一层
        x = self.act(x)                 # 激活
        x = self.fc2(x)                 # 流过输出层
        return x

model = MLP()#创建模型实例
criterion = nn.CrossEntropyLoss()#交叉熵
loss = criterion(logits, yb)     # logits: (32,2)，yb: (32,)


