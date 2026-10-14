'''
lstm = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
#input_size=2：告诉 LSTM，“每个时刻，你会收到 2 个数字（比如温度和湿度）”。
#hidden_size=32：告诉 LSTM，“你的记忆容量是 32 维的向量”。（指定输出特征维度为 32）。
#num_layers=1：只搭 1 层 LSTM。如果是 2，就是第一层的输出喂给第二层。
x = torch.randn(64, 24, 2)                 # 64 个样本，每个用过去 24 个时刻、每刻 2 个指标
#(Batch_Size, Sequence_Length(时间步长（序列长度）), Input_Size(特征维度。每个时刻包含了 2 个指标))
out, (h_n, c_n) = lstm(x)#out：所有时刻的隐藏状态的集合***(h_n, c_n)：最后一个时刻的隐藏状态和细胞状态
print(out.shape)                           # (64, 24, 32)
print(h_n.shape)                           # (1, 64, 32)
'''
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
'''
def make_windows(series, lookback, horizon):
    #series：原始的连续数据。形状是 (总时长, F(测的指标个数))
    #lookback：回看窗口大小（用过去多少步来预测）
    #horizon：预测窗口大小（预测未来多少步）
    X, y = [], []#装输入和标签的空列表
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])                       # 过去 lookback 步
        y.append(series[i + lookback : i + lookback + horizon])  # 未来 horizon 步
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 例：每小时一条记录，用过去 24 小时预测未来 7 天（168 小时）
X, y = make_windows(series, lookback=24, horizon=168)
#X 的形状将是：(len(series) - lookback - horizon + 1, lookback, F)
#y 的形状将是:(len(series) - lookback - horizon + 1, horizon, F)
n = int(0.8 * len(X))
X_train, X_val = X[:n], X[n:]     # 前 80% 时间训练，后 20% 验证
y_train, y_val = y[:n], y[n:]
'''
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, horizon=168):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon * input_size)  # 一次性输出未来全部值

    def forward(self, x):                       # x的形状: (N, lookback, F)
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]                    # 取最后一个时间步的记忆 (N, hidden)
        y = self.head(last)                     # (N, horizon*F)
        return y.view(-1, self.horizon, self.input_size)  #其中 -1 表示让 PyTorch 自动推断第一维大小，还原成 (N, horizon, F)，.view等价于.reshape

model = LSTMForecaster(input_size=2, hidden_size=32, horizon=168)
pred = model(torch.randn(64, 24, 2))
print(pred.shape)                               # (64, 168, 2)：未来168小时的温度与湿度

criterion = nn.MSELoss()                        # 回归任务（第 2 部分配套表）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#model.parameters()：告诉优化器要更新哪些参数

for epoch in range(50):
    model.train()
    for xb, yb in train_loader:                 # xb:(N,24,2)输入，过去 24 小时, yb:(N,168,2)标签，真实未来 168 小时
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()                         #计算所有参数的梯度
        optimizer.step()
