import torch
from torch import nn                          # 神经网络相关工具
from torch.utils.data import Dataset, DataLoader, TensorDataset
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x        # 前向：y 是 x 的函数，过程被记录成计算图
y.backward()              # 反向传播：自动求导
print(x.grad)             # 查看 x 的梯度：dy/dx = 2x+3，在 x=2 处为 7 → tensor(7.)
x.grad = None# 直接清空（简单写法）
'''
做预测、验证时我们不需要梯度，建图纯属浪费内存和时间：
with torch.no_grad():
    pred = model(x)      # with里的计算内容不会被记录，省内存、速度更快
'''

#数据满足y = 3x（加一点噪声），我们让 PyTorch 自己把斜率 `w` 学出来：
torch.manual_seed(0)#设置随机种子
x=torch.linspace(-1,1,100)
x_data = torch.linspace(-1, 1, 100)
y_data = 3 * x_data + 0.1 * torch.randn(100)
w = torch.tensor(0.0, requires_grad=True)#创建一个标量张量，初始值为 0.0
lr = 0.1                                     # 学习率：每步迈多大
for step in range(50):
    pred = w * x_data                        # 用当前的w计算预测值
    loss = ((pred - y_data) ** 2).mean()     # 均方误差损失,loss 是一个标量张量
    if w.grad is not None:                   #第一次循环时 w.grad 是 None，所以用 if 判断
        w.grad=None                          #清空上一轮梯度
    loss.backward()                          #反向传播：计算损失对w的梯度，计算结果会存入 w.grad 中
    with torch.no_grad():                    # 更新参数本身不需要被记录
        w -= lr * w.grad                     # 沿负梯度方向走一步

print(w.item())     # 约等于 3.0,把标量张量转成 Python 浮点数


