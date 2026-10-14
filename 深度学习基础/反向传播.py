import torch
xs = torch.tensor([1.0, 2.0, 3.0, 4.0])
ys = torch.tensor([3.0, 5.0, 7.0, 9.0])

# —— 纯手动：梯度公式自己写（这就是反向传播在干的活）——
w = torch.tensor(0.0); b = torch.tensor(0.0); lr = 0.05
for _ in range(200):
    pred = w * xs + b
    grad_w = (2 * (pred - ys) * xs).mean()   # ∂L/∂w
    grad_b = (2 * (pred - ys)).mean()        # ∂L/∂b
    w -= lr * grad_w                          # 手动更新
    b -= lr * grad_b
print(w.item(), b.item())                     # 约 2.0 和 1.0

# —— PyTorch：backward 自动完成上面两行梯度计算 ——
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.05)
for _ in range(200):
    opt.zero_grad()
    loss = ((w * xs + b - ys) ** 2).mean()
    loss.backward()        # 自动反向传播，w.grad、b.grad 被填好
    opt.step()             # 优化器自动更新
print(w.item(), b.item())                 # 同样约 2.0 和 1.0
#.item()：从一个只包含一个元素的 PyTorch 张量中，取出那个元素，转成普通的 Python 数字
