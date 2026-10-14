import numpy as np
lst1=[1,2,3]
lst2=[4,5,6]
lst3=[7,8,9]
a=np.array(lst1)#创建一个一维数组
print(a)
b=np.array([lst1,lst2])#创建一个二维数组，也是一个2行3列矩阵
print(b)
print('数组形状：',b.shape)
print('数组类型：',b.dtype)
print('数组元素个数：',b.size)
c=b.reshape(3,2)
print(c)
time_axis=np.arange(0,10,0.1)#起点, 终点(不含), 步长
print(time_axis)
time_axis1=np.linspace(0,10,100)#起点, 终点(包含), 分割成100个点=99个空
print(time_axis1)
a=np.arange(12)#生成0-11的一维数组
print(a)
b=a.reshape(2,6)#把这12个数字，强行折叠成2行6列的二维矩阵
# 注意：变换前后的元素总数必须一致，12 = 2 * 6，否则会报错。
print(b)
a=np.array([[1,2,3],[4,5,6]])
r=a+5
#广播机制把5自动拉伸成了[[5, 5, 5],[5, 5, 5]]，然后对应位置相加。
print(r)
zeros_arr = np.zeros((3, 3)) # 3行3列的全0矩阵
ones_arr = np.ones((2, 2))   # 2行2列的全1矩阵
eye_arr = np.eye(3)#3阶单位对角阵
diag_arr=np.diag([1,3,2,4])#对角阵，里面的元素为1，3，2，4
print(diag_arr)
print(eye_arr)
print(zeros_arr)
print(ones_arr)
print('生成的随机数组：',np.random.random(10))#随机生成10个随机数
print('生成的随机数组：',np.random.rand(3,4))#生成服从均匀分布的随机数（3行4列）
print('生成的随机数组：',np.random.randn(3,4))#生成服从正态分布的随机数（3行4列）
print('生成的随机数组：\n',np.random.randint(0,100,size=[3,4]))#生成一个随机数组，数值范围在0-100，且是一个3行4列的矩阵
x=np.arange(10)
print(x)
print('索引为5的元素是：',x[5])
print(x[0:7])#不包括索引是7的那个元素
x[1:3]=10,11#将索引是1，2的元素替换为10，11
print(x)
print(x[6:0:-2])#步长为负数时，开始索引要小于结束索引，且负数步长=倒着取，结束索引仍然是不取的
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr)
print(arr[0,1:3])#取索引为0的第1行，的第2列和第3列的元素
#行与列之间用'，'隔开
print(arr[1:3,0:2])
print(arr[:,1:2])#取所有行的第2列元素
print(arr[0,2])
print(arr[arr>5])#取出矩阵里所有大于 5 的元素

