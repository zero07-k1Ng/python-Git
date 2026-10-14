import torch
from torch import nn

# 回归
nn.MSELoss()(torch.tensor(1.5), torch.tensor(2.0))      # tensor(0.2500)

# 多分类：CrossEntropyLoss 内部已包含 Softmax，喂"原始分数 logits + 整数标签"即可
logits = torch.tensor([[2.0, 1.0, 0.1]])   # 模型直接输出，不要再手动 softmax
target = torch.tensor([0])                 # 真实类别编号
loss = nn.CrossEntropyLoss()(logits, target)

# 二分类推荐 BCEWithLogitsLoss（内部融合 Sigmoid，数值更稳定）
# nn.BCEWithLogitsLoss()(logit, target_float)