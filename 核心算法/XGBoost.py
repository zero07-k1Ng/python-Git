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

# ===================== XGBoost分类器 =====================
# 核心参数说明：
# n_estimators / learning_rate / max_depth：和GBDT一致
# gamma：节点分裂所需的最小损失下降，越大越保守，控制过拟合
# reg_alpha / reg_lambda：L1/L2正则项，XGBoost的核心改进之一
# subsample / colsample_bytree：样本/特征采样
# objective：目标函数，二分类用'binary:logistic'
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    gamma=0.1,             #节点分裂所需的最小损失下降
#在决策树中，一个节点要不要继续分裂成两个子节点，要看分裂后损失能降低多少。
#如果降低的量小于 gamma，就不分裂。
#gamma 越大，分裂条件越严格，树越保守，越简单。
#gamma 越小，越容易分裂，树越复杂，可能过拟合。
#所以 gamma 是一种正则化手段，控制树的生长。
    reg_alpha=0.1,         # L1正则
    reg_lambda=1.0,        # L2正则
    subsample=0.8,         #样本采样比例
    colsample_bytree=0.8,  # 特征采样:每棵树训练时，从所有特征中随机抽取 80% 的特征
    #XGBoost 是每棵树抽一次，而不是每个节点抽。
    objective='binary:logistic',#目标函数，也就是模型要优化的损失函数。
    #'binary:logistic'：二分类任务，使用逻辑回归损失，输出概率
    eval_metric='logloss',#评估指标：'logloss' 是对数损失，适合二分类概率输出
    use_label_encoder=False,#XGBoost 旧版本的一个参数，无用
    random_state=42,
    n_jobs=-1
)

# 训练
start = time()
xgb_model.fit(X_train, y_train, verbose=False)#verbose=False：训练时不打印详细过程
xgb_train_time = time() - start

# 预测与评估
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

print("===== XGBoost 评估结果 =====")
print(f"训练集准确率：{accuracy_score(y_train, y_train_pred):.4f}")
print(f"测试集准确率：{accuracy_score(y_test, y_test_pred):.4f}")
print(f"训练耗时：{xgb_train_time:.4f}秒")

# 特征重要性
feature_importance_xgb = pd.DataFrame({
    '特征': X.columns,
    '重要性': xgb_model.feature_importances_
}).sort_values('重要性', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance_xgb)
plt.title('XGBoost 特征重要性Top10')
plt.tight_layout()
plt.show()
