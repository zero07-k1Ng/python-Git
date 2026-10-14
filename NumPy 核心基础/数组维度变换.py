import numpy as np
t=np.linspace(0,10,100)
print(t.shape)#运行结果为：(100,)为一维向量
t1=t.reshape(-1,1)#这里的-1是“自动计算”的意思，NumPy自动帮你算行数
print(t1)
#一维数组既不是行向量也不是列向量
#做运算前经常需要reshape成列向量/行向量，不然会报错。

print(t1.T)#列向量转置成行向量，shape从(100,1)变成(1,100)

t2=t1.squeeze()#去掉长度为1的维度，
print(t2)#把(100,1)的列向量变回(100,)的一维数组
arr = np.array([1,2,3])#广播规则
print(arr + 1)  # 每个元素加1，输出[2,3,4]
print(arr * 2)  # 每个元素乘2，输出[2,4,6]
'''
你有 100 组 3 通道传感器数据data.shape=(100,3)
现在要给每个通道加校准值：温度 + 0.5℃，湿度 - 0.3%，光照 + 0.2
校准值是bias = np.array([0.5, -0.3, 0.2])（shape=(3,)）
执行：new_data = data + bias
NumPy 自动把 bias 复制 100 行，变成 100x3 的矩阵，然后逐元素相加
'''
x=np.random.randint(0,100,size=[10,3])
print(x)
print('-'*40)
bias = np.array([0.5, -0.3, 0.2])
new_x=bias+x
print(new_x)
