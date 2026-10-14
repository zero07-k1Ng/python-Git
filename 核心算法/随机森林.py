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

# ===================== 随机森林分类器 =====================
# 核心参数说明：
# n_estimators：树的数量，通常100-1000，越多越稳定但速度越慢
# max_features：节点分裂时随机抽取的特征数，是随机森林的核心
# bootstrap：是否有放回抽样（Bagging的核心）
# n_jobs：并行训练的CPU核心数，-1表示使用全部核心
rf_model = RandomForestClassifier(
    n_estimators=100,      # 100棵决策树：树越多，模型通常越稳定，方差越小
    max_depth=5,           # 每棵树的最大深度
    max_features='sqrt',   # 每次节点分裂时，随机选sqrt(总特征数)个特征，体现特征随机性
    #例如：总共有16个特征，sqrt(16)=4，每次分裂只随机看 4 个特征
    bootstrap=True,        # 开启有放回抽样（Bagging）
    n_jobs=-1,             # 并行训练，使用所有可用的 CPU 核心
    random_state=42
)

# 训练
start = time()
rf_model.fit(X_train, y_train)
rf_train_time = time() - start

# 预测与评估
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

print("===== 随机森林 评估结果 =====")
print(f"训练集准确率：{accuracy_score(y_train, y_train_pred):.4f}")
print(f"测试集准确率：{accuracy_score(y_test, y_test_pred):.4f}")
print(f"训练耗时：{rf_train_time:.4f}秒")

# 特征重要性
feature_importance_rf = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance_rf)
plt.title('随机森林 特征重要性Top10')
plt.tight_layout()
plt.show()