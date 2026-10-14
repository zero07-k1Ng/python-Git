# ===================== 1. 导入所有依赖库 =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# 数据集与数据划分
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 算法库
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 决策树
from sklearn.ensemble import RandomForestClassifier        # 随机森林
from sklearn.ensemble import GradientBoostingClassifier    # GBDT
from xgboost import XGBClassifier                          # XGBoost
from lightgbm import LGBMClassifier                        # LightGBM

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 加载并划分数据 =====================
# 加载乳腺癌数据集（二分类任务：预测肿瘤是良性/恶性）
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  # 特征
y = data.target                                           # 标签：0=恶性，1=良性

# 划分训练集70%、测试集30%，random_state保证结果可复现
np.random.seed(42)
#train_test_split:把 X 和 y 随机拆成两部分：一部分用于训练模型，一部分用于测试模型
#stratify=y 表示：按照标签y的比例进行分层抽样
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y
)
print(f"训练集样本数：{X_train.shape[0]}，特征数：{X_train.shape[1]}")
print(f"测试集样本数：{X_test.shape[0]}")
print(f"类别分布：良性{sum(y==1)}，恶性{sum(y==0)}")

# ===================== GBDT分类器 =====================
# 核心参数说明：
# n_estimators：树的数量（迭代次数）
# learning_rate：学习率，控制每棵树的贡献，越小需要越多的树
#最终预测=初始预测+第1棵树输出×学习率+第2棵树输出×学习率+⋯
# max_depth：每棵树的深度，GBDT通常用浅树（3-8层）
gbdt_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,       # GBDT通常用浅树，靠数量提升精度
    subsample=0.8,     # 每次迭代训练一棵新树时，只随机使用 80% 的样本
    random_state=42
)

# 训练（串行训练，无法并行）
start = time()
gbdt_model.fit(X_train, y_train)
gbdt_train_time = time() - start

# 预测与评估
y_train_pred = gbdt_model.predict(X_train)
y_test_pred = gbdt_model.predict(X_test)

print("===== GBDT 评估结果 =====")
print(f"训练集准确率：{accuracy_score(y_train, y_train_pred):.4f}")
print(f"测试集准确率：{accuracy_score(y_test, y_test_pred):.4f}")
print(f"训练耗时：{gbdt_train_time:.4f}秒")

# 特征重要性
feature_importance_gbdt = pd.DataFrame({
    '特征': X.columns,
    '重要性': gbdt_model.feature_importances_
}).sort_values('重要性', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance_gbdt)
plt.title('GBDT 特征重要性Top10')
plt.tight_layout()
plt.show()
