import numpy as np

# ===================== 1. 向量基础运算 =====================
# 定义两个向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print("向量加法:", v1 + v2)
print("向量数乘:", 2 * v1)
print("向量点积（内积）:", np.dot(v1, v2))  # 对应元素相乘再相加，线性回归核心运算
print("向量L2范数（模长）:", np.linalg.norm(v1))  # KNN、正则化中常用
print("向量对应元素相乘:", v1 * v2)


# ===================== 2. 矩阵基础运算 =====================
# 定义矩阵
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("\n矩阵加法:\n", A + B)
print("矩阵数乘:\n", 3 * A)
print("矩阵乘法（点积）:\n", np.dot(A, B))  # 满足：前行乘后列
print("矩阵转置:\n", A.T)
print("矩阵的逆:\n", np.linalg.inv(A))  # 线性回归闭式解会用到
print("矩阵行列式:", np.linalg.det(A))

# ===================== 3. 特征值与特征向量 =====================
# 定义：若 A·v = λ·v，则λ是特征值，v是对应特征向量
# 常用于 PCA、谱聚类、矩阵分解等算法
eigenvalues, eigenvectors = np.linalg.eig(A)
#在返回的 eigenvectors 二维数组中，特征向量是按列排列的
print("\n矩阵A的特征值:", eigenvalues)
print("矩阵A的特征向量（每一列是一个特征向量）:\n", eigenvectors)

# 验证：A·v = λ·v
for i in range(len(eigenvalues)):
    Av = np.dot(A, eigenvectors[:, i])
    lambda_v = eigenvalues[i] * eigenvectors[:, i]
    print(f"第{i+1}个特征值验证：", np.allclose(Av, lambda_v))
