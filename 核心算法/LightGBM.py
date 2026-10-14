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

# ===================== LightGBM分类器 =====================
# 核心参数说明：
# num_leaves：叶子节点数，Leaf-wise生长的核心参数，替代max_depth
# max_depth：限制树深，防止过拟合
# learning_rate / n_estimators：和XGBoost一致
# boosting_type：'gbdt'为传统梯度提升
# feature_fraction：特征采样，对应colsample_bytree
# bagging_fraction / bagging_freq：样本采样
# categorical_feature：指定类别特征，LightGBM原生支持类别特征
lgb_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,        #每棵树最多有 31 个叶子节点
    max_depth=5,          # 限制深度，防止过拟合
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='binary',
    metric='binary_logloss',
    random_state=42,
    n_jobs=-1,
    verbose=-1#verbose=-1：不输出任何训练日志，保持安静
)

# 训练
start = time()
lgb_model.fit(X_train, y_train)
lgb_train_time = time() - start

# 预测与评估
y_train_pred = lgb_model.predict(X_train)
y_test_pred = lgb_model.predict(X_test)

print("===== LightGBM 评估结果 =====")
print(f"训练集准确率：{accuracy_score(y_train, y_train_pred):.4f}")
print(f"测试集准确率：{accuracy_score(y_test, y_test_pred):.4f}")
print(f"训练耗时：{lgb_train_time:.4f}秒")

# 特征重要性
feature_importance_lgb = pd.DataFrame({
    '特征': X.columns,
    '重要性': lgb_model.feature_importances_
}).sort_values('重要性', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance_lgb)
plt.title('LightGBM 特征重要性Top10')
plt.tight_layout()
plt.show()

