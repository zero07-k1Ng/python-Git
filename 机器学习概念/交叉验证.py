import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold

# 准备数据
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=4)
model = LogisticRegression()

# ===================== 1. 5折交叉验证 =====================
# cv=5 表示将数据分成5份，轮流做验证集
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
#model：要评估的模型，cv=5：5折交叉验证，scoring='accuracy'：评价指标是准确率，也就是预测正确的比例

print("5折交叉验证每折准确率:", scores.round(4))
print(f"平均准确率: {scores.mean():.4f}")
print(f"标准差: {scores.std():.4f}")

# ===================== 2. 手动实现K折 =====================
#创建一个 K 折划分器
kf = KFold(n_splits=5, shuffle=True, random_state=42)#n_splits=5：分成 5 折
#shuffle=True：在划分前先把数据打乱
manual_scores = []

for train_idx, val_idx in kf.split(X):#split(X)主要用 X 的长度，会生成 5 组索引
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)#用验证集评估模型，分类问题默认返回准确率。
    manual_scores.append(score)

print("\n手动5折交叉验证每折准确率:", np.round(manual_scores, 4))#保留 4 位小数
print(f"手动平均准确率: {np.mean(manual_scores):.4f}")
