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





# ===================== 决策树分类器 =====================
# 核心参数说明：
# max_depth：树的最大深度，控制过拟合，None表示不限深度
# min_samples_split：节点分裂所需的最少样本数
# min_samples_leaf：叶子节点最少样本数
# criterion：分裂准则，分类用'gini'（基尼系数）或'entropy'（信息熵）
dt_model = DecisionTreeClassifier(
    max_depth=3,          # 限制树深，防止过拟合：从根节点到叶子节点最多经过几次分裂
    criterion='gini',     # 使用基尼系数做分裂准则：决策树每次分裂时，要选择一个特征和阈值，让子节点尽可能“纯”，纯”的意思是：一个节点里的样本尽量属于同一类，基尼系数越小，节点越纯
    min_samples_leaf=5,   # 叶子节点最少5个样本：如果某个节点分裂后，某个子节点样本数少于 5，就不允许这样分裂
    random_state=42
)

# 训练模型
start = time()#记录训练开始的时间
dt_model.fit(X_train, y_train)
dt_train_time = time() - start

# 预测
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# 评估
print("===== 决策树 评估结果 =====")
print(f"训练集准确率：{accuracy_score(y_train, y_train_pred):.4f}")
print(f"测试集准确率：{accuracy_score(y_test, y_test_pred):.4f}")
print(f"训练耗时：{dt_train_time:.4f}秒")
print("\n分类报告：")
print(classification_report(y_test, y_test_pred))
#support：表示该类在 ‘真实测试集’ 中有多少个样本
#weighted avg：对每个类别的指标按样本数加权平均。样本多的类别权重大，样本少的类别权重小。它更接近整体表现，因为考虑了类别不平衡
# 画出决策树结构，直观理解"层层判断"的逻辑
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,#要画的决策树模型
    feature_names=data.feature_names,
    class_names=['恶性', '良性'],#类别名称
    filled=True,#给节点填充颜色，颜色越深表示该节点越纯
    rounded=True,#节点边框用圆角，看起来更美观
    fontsize=11#图中字体大小
)
plt.title("决策树结构可视化（深度=3）")
plt.show()

# 提取特征重要性并排序
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': dt_model.feature_importances_
}).sort_values('重要性', ascending=False).head(10)#按“重要性”列从大到小排序，head(10)：取前 10 行

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance)#画条形图
plt.title('决策树 特征重要性Top10')
plt.tight_layout()
plt.show()

# 对比不同深度的效果
depth_list = [1, 3, 5, 10, None]
train_acc = []
test_acc = []

for depth in depth_list:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

# 绘图
plt.figure(figsize=(10, 6))
plt.plot([str(d) for d in depth_list], train_acc, marker='o', label='训练集准确率')
plt.plot([str(d) for d in depth_list], test_acc, marker='s', label='测试集准确率')#方形标记每个数据点
#[str(d) for d in depth_list]：用来生成一个新列表，遍历 depth_list 中的每一个元素 d，对每个 d 调用 str(d)，把结果放到一个新列表里
plt.xlabel('树的最大深度')
plt.ylabel('准确率')
plt.title('决策树深度与过拟合的关系')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("结论：树越深，训练集准确率越高，但测试集准确率先升后降，出现过拟合。")
print("工业界通常通过限制max_depth、min_samples_leaf等参数做预剪枝。")