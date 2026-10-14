import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# ===================== 1. 准备数据 =====================
X, y = make_classification(n_samples=200, n_features=4, n_informative=2,
                           n_redundant=0, random_state=42)


# ===================== 2. 手动实现高斯朴素贝叶斯 =====================
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)#np.unique(y)：去掉数组 y 中的重复值，返回其中所有唯一值，并且默认会从小到大排序。
        #self.classes：所有类别，比如 [0, 1]。
        self.mean = {}#self.mean：保存每个类别下每个特征的均值。
        self.var = {}
        self.prior = {}#self.prior：保存每个类别的先验概率。

        for c in self.classes:
            X_c = X[y == c]#取出所有属于类别 c 的样本
            self.mean[c] = np.mean(X_c, axis=0)  # 每个特征的均值
            self.var[c] = np.var(X_c, axis=0)  # 每个特征的方差
            self.prior[c] = len(X_c) / len(X)  # 当前类别样本数占总样本数的比例

    def _gaussian_pdf(self, x, mean, var):
        """高斯概率密度函数"""
        eps = 1e-6
        return (1 / np.sqrt(2 * np.pi * (var + eps))) * np.exp(-(x - mean) ** 2 / (2 * (var + eps)))

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}#保存每一个样本x属于每个类别的“分数”。
            for c in self.classes:
                # 对数后验概率 = 对数先验 + 对数似然
                log_prior = np.log(self.prior[c])
                log_likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[c], self.var[c])))
                posteriors[c] = log_prior + log_likelihood#未归一化的对数后验
            pred = max(posteriors, key=posteriors.get)#会返回字典中值最大的那个键，也就是对数后验最大的类别。
            predictions.append(pred)#把这个类别加入预测结果。
        return np.array(predictions)


# 测试手动实现
gnb_manual = GaussianNaiveBayes()
gnb_manual.fit(X, y)
y_pred_manual = gnb_manual.predict(X)
print(f"手动实现高斯朴素贝叶斯准确率: {accuracy_score(y, y_pred_manual):.4f}")

# ===================== 3. sklearn 实现 =====================
gnb_sk = GaussianNB()
gnb_sk.fit(X, y)
y_pred_sk = gnb_sk.predict(X)
print(f"sklearn 高斯朴素贝叶斯准确率: {accuracy_score(y, y_pred_sk):.4f}")
