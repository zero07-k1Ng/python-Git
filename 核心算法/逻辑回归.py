import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 准备二分类数据 =====================
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
y = y.reshape(-1, 1)

#逻辑回归是预测“是/否”这类类别
# ===================== 2. 手动实现逻辑回归 =====================
def sigmoid(z):
    """sigmoid激活函数：将任意值映射到(0,1)"""
    return 1 / (1 + np.exp(-z))
# sigmoid输出 0.9，表示“有90%的概率是恶性”


def logistic_regression_gd(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置
    theta = np.zeros((n + 1, 1))  # 初始化参数

    for epoch in range(epochs):
        z = X_b.dot(theta)
        y_pred = sigmoid(z)
        # 交叉熵损失的梯度
        gradients = 1 / m * X_b.T.dot(y_pred - y)
        theta = theta - lr * gradients
    return theta


theta_lr = logistic_regression_gd(X, y, lr=0.5, epochs=2000)
print("手动实现逻辑回归参数：")
print(f"截距b = {theta_lr[0][0]:.4f}")
print(f"权重w1 = {theta_lr[1][0]:.4f}, w2 = {theta_lr[2][0]:.4f}")

# 手动预测
X_b_test = np.c_[np.ones((len(X), 1)), X]
y_prob_manual = sigmoid(X_b_test.dot(theta_lr))
y_pred_manual = (y_prob_manual >= 0.5).astype(int)
print(f"手动实现准确率: {accuracy_score(y, y_pred_manual):.4f}")

# ===================== 3. sklearn 实现 =====================
model_lr = LogisticRegression()
model_lr.fit(X, y.ravel())#y.ravel()：你之前把 y 变成了 (200, 1) 的二维列向量。但 sklearn 的 fit 通常期望标签是一维数组 (200,)
#y.ravel():把 (200, 1) 压扁成 (200,)
y_pred_sk = model_lr.predict(X)
#返回的是每个样本的类别标签（0 或 1），不是概率。
print(f"\nsklearn 逻辑回归准确率: {accuracy_score(y, y_pred_sk):.4f}")#y_pred_sk 是预测标签

# 可视化决策边界
#1.确定网格范围
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#2.生成网格点
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),#生成网格坐标矩阵
                       np.arange(x2_min, x2_max, 0.02))
#3.展平网格并预测
Z = model_lr.predict(np.c_[xx1.ravel(), xx2.ravel()])#xx1.ravel()把 (300, 300)的二维数组拉成 (90000,)的一维数组。
#对这 90000 个点逐一预测类别，返回 (90000,) 的数组 Z，每个值是 0 或 1。
Z = Z.reshape(xx1.shape)#把(90000,)的预测结果重新变回 (300, 300) 的二维数组

plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.Paired)
#contourf的意思是 “填充等高线”,在这里，它根据每个网格点的预测类别 Z（0 或 1）填充颜色。
#cmap=plt.cm.Paired：使用的配色方案。Paired 是一套成对出现的颜色
#整个平面被涂成两种颜色,一种代表模型预测的类别 0 区域，另一种代表类别 1 区域。两种颜色的交界线，就是决策边界
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='black', cmap=plt.cm.Paired)
#c=y.ravel()：点的颜色由真实标签决定。类别 0 和类别 1 用不同颜色表示
plt.title('逻辑回归决策边界')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.show()
