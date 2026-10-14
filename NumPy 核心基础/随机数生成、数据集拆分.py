import numpy as np
a=np.random.seed(42)  # 42是行业惯例，随便填个整数就行，加了这个每次随机数都一样
print(a)
# 假设data是你的100组实验数据
data = np.random.randn(100,3)
# 第一步：生成打乱的索引
indices = np.random.permutation(len(data))
#len(data) 返回数据行数，这里是100。
#np.random.permutation(100)会返回一个包含 0 到 99 的随机排列数组
#例如 [37, 12, 89, 5, ...]
# 第二步：80%当训练集，20%当测试集
split = int(0.8 * len(data))
#split作为切分索引的分界点。
train_idx = indices[:split]
test_idx = indices[split:]
# 第三步：拆分数据
train_data = data[train_idx]
test_data = data[test_idx]
print(train_data)
print(test_data)