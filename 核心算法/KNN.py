import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 准备数据 =====================
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)


# ===================== 2. 手动实现KNN =====================
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # 计算与所有训练样本的欧氏距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))#axis=1：按行求和
            # 取最近的k个索引
            k_indices = distances.argsort()[:self.k]#argsort()：返回数组排序后的索引。
            # 比如 [3, 1, 4] 的 argsort 是 [1, 0, 2]，表示最小值在原数组的第 1 位
            #[:self.k]：取排序后的前 k 个索引，也就是距离最近的 k 个邻居的索引
            # 投票
            k_labels = self.y_train[k_indices]#取出这 3 个邻居的类别标签
            pred = np.bincount(k_labels).argmax()#统计每个类别出现的次数
            # 比如 k_labels = [0, 0, 1]，bincount 会返回 [2, 1]，表示类别 0 出现 2 次，类别 1 出现 1 次
            predictions.append(pred)
        return np.array(predictions)#循环结束后，把列表转成 numpy 数组返回。


# 测试手动实现
knn_manual = KNNClassifier(k=3)
knn_manual.fit(X, y)
y_pred_manual = knn_manual.predict(X)
print(f"手动实现KNN(k=3)准确率: {accuracy_score(y, y_pred_manual):.4f}")

# ===================== 3. sklearn 实现 =====================
knn_sk = KNeighborsClassifier(n_neighbors=3)
knn_sk.fit(X, y)
y_pred_sk = knn_sk.predict(X)
print(f"sklearn KNN(k=3)准确率: {accuracy_score(y, y_pred_sk):.4f}")

# 可视化决策边界
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1#X[:, 0] 是所有样本的第一个特征(横坐标)
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1#X[:, 1] 是所有样本的第二个特征(纵坐标)
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))
#np.arange(x1_min, x1_max, 0.02)：从 x1_min 到 x1_max，每隔 0.02 取一个点。
#np.meshgrid(...)：生成网格坐标矩阵
Z = knn_sk.predict(np.c_[xx1.ravel(), xx2.ravel()])
#xx1.ravel()：把 (300, 300) 的二维数组拉成 (90000,) 的一维数组
Z = Z.reshape(xx1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.Paired)
#contourf 的意思是 “填充等高线”，在这里，它根据每个网格点的预测类别 Z（0 或 1）填充颜色。
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
#cmap=plt.cm.Paired：使用的配色方案。Paired 是一套成对出现的颜色
#两种颜色的交界线，就是决策边界。
plt.title('KNN决策边界 (k=3)')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.show()
