import numpy as np
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(A)
print(B)
'''
print(A + B)  # 对应位置相加：[[6,8],[10,12]]
print(A * 2)  # 每个元素乘2：[[2,4],[6,8]]
print(A * B)  # 对应位置相乘（哈达玛积）：[[5,12],[21,32]]
'''
arr=np.array([[1,2,3],[4,5,6]])
a=np.mean(arr,axis=0)#求均值(如果不指定axis参数，np.mean就是求全局均值)
#axis=0：按列计算,行数没了
#axis=1：按行计算，列数没了
#axis等于几，算完那个维度就消失了。原来shape是(100,3)
#axis=0算完shape是(3,)，axis=1算完shape是(100,)
print(a)
b=np.std(arr)#求标准差
print(b)
c=np.var(arr)#方差
print(c)
'''
np.max(arr)/np.min(arr)#求最大/最小值
np.argmax(arr)/np.argmin(arr)#找最大/最小值的位置
np.sum(arr)#求和	

print(A@B)#矩阵乘法
print(np.linalg.inv(A))#逆矩阵
print(np.linalg.matrix_rank(A))#矩阵的秩
print(np.linalg.solve(A,b))#解线性方程组Ax=b	
'''
