import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
'''
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)        # 最基础的梯度下降
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)       # 自适应学习率，实战最常用
#`optimizer.zero_grad()`：清空所有参数上一轮累积的梯度
#`optimizer.step()`：按梯度更新所有参数
'''
#训练前调 `model.train()`，验证前调 `model.eval()`
'''
torch.save(model.state_dict(), "model.pth")         #把 state_dict 序列化并保存到文件 model.pth
model2 = MLP()                                        # 先按原结构造一个空模型
model2.load_state_dict(torch.load("model.pth"))       #从文件读取之前保存的 state_dict，返回一个字典，默认会加载到 CPU
model2.eval()
'''
N=1000
X=torch.randn(N,10)
W_true=torch.randn(10,2)
y=(X@W_true).argmax(dim=1)

full_ds=TensorDataset(X,y)
n_train=int(0.8*len(full_ds))
train_ds,val_ds=random_split(full_ds,[n_train,len(full_ds)-n_train])
train_loader=DataLoader(train_ds,batch_size=32,shuffle=True)
val_loader=DataLoader(val_ds,batch_size=32,shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.neet=nn.Sequential(
            nn.Linear(10,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
    def forward(self,x):
        return self.neet(x)
model=MLP().to(device)
c=nn.CrossEntropyLoss()
yhq=torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss=0
    for xb,yb in train_loader:
        xb,yb=xb.to(device),yb.to(device)
        yhq.zero_grad()
        log=model(xb)
        loss=c(log,yb)
        loss.backward()
        yhq.step()
        total_loss+=loss.item()*xb.size(0)
    total_train_loss=total_loss/len(train_ds)
    model.eval()
    correct=0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb=xb.to(device),yb.to(device)
            pred=model(xb).argmax(dim=1)#沿着第 1 个维度，也就是“类别维度”，找最大值所在的索引
            correct+=(pred==yb).sum().item()
        val_acc=correct/len(val_ds)
        print(f"Epoch {epoch + 1:02d} | 训练损失 {total_train_loss:.4f} | 验证准确率 {val_acc:.3f}")







