import torch

m = torch.randn(1, 3, 1)
print(m.shape)                 # torch.Size([1, 3, 1])
print(m.squeeze().shape)       # torch.Size([3])

x = torch.randn(3)
print(x.unsqueeze(0).shape)    # torch.Size([1, 3])

img = torch.randn(64, 64, 3)
img_chw = img.permute(2, 0, 1)
print(img_chw.shape)           # torch.Size([3, 64, 64])

import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)   # NumPy → Tensor（共享内存，改一个另一个也变）
t2 = torch.tensor(arr)      # 这种写法会复制一份，更安全
back = t.numpy()            # Tensor → NumPy（CPU 上才行）
