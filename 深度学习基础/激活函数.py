import torch
import torch.nn.functional as F

z = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
print(torch.sigmoid(z))   # 全部压到 0~1
print(torch.relu(z))      # [-2,-0.5] 变 0，其余保留 → [0,0,0,1,3]
print(F.softmax(torch.tensor([2.0, 1.0, 0.1]), dim=0))  # 三个和为1的概率
