'''
带动量 SGD:
v ← β·v + grad        # β 常取 0.9
w ← w − lr·v

Adam:
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
`weight_decay` 即 L2 正则化,在损失里对大权重施加惩罚，抑制参数过大，有助于缓解过拟合
'''
#optimizer.zero_grad()   # 1. 清空上一批累积的梯度
#loss.backward()         # 2. 反向传播，算出本批梯度
#optimizer.step()        # 3. 按梯度更新全部参数
