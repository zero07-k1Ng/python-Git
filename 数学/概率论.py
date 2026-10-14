import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 期望与方差 =====================
data = np.array([1, 2, 3, 4, 5])

mean = np.mean(data)       # 期望（均值）
variance = np.var(data)    # 方差
std = np.std(data)         # 标准差

print("数据:", data)
print("期望（均值）:", mean)
print("方差:", variance)
print("标准差:", std)

# ===================== 2. 贝叶斯公式 =====================
# 公式：P(A|B) = P(B|A) * P(A) / P(B)
# 例子：某种疾病发病率P(A)=0.001，检测阳性率
# 患病者阳性P(B|A)=0.99，健康者阳性P(B|¬A)=0.05
# 求：检测阳性时，真正患病的概率 P(A|B)

P_A = 0.001          # 先验概率：患病概率
P_notA = 1 - P_A     # 健康概率
P_B_given_A = 0.99   # 似然：患病时检测阳性
P_B_given_notA = 0.05# 似然：健康时检测阳性

# 全概率公式
#P(B)=P(B∣A)P(A)+P(B∣¬A)P(¬A)
P_B = P_B_given_A * P_A + P_B_given_notA * P_notA

# 贝叶斯公式
P_A_given_B = (P_B_given_A * P_A) / P_B

print(f"\n检测阳性时真正患病的概率: {P_A_given_B:.4f}")
print("这也是朴素贝叶斯分类器的核心公式")

# ===================== 3. 最大似然估计（MLE） =====================
# 目标：已知样本，估计分布的参数
# 例：样本服从正态分布N(μ,σ²)，用MLE估计μ和σ
np.random.seed(42)
samples = np.random.normal(loc=5, scale=2, size=1000)  # 真实μ=5, σ=2
#np.random.normal:从正态分布中抽取随机样本
#loc=5：正态分布的均值=5,分布的中心在 5
#scale=2：正态分布的标准差=2,注意不是方差，方差是标准差的平方=4
#size=1000：抽取 1000 个样本，返回一个长度为 1000 的一维数组
samples=np.random.normal(loc=5,scale=2,size=500)

# MLE估计：均值=样本均值，标准差=样本标准差
mu_mle = np.mean(samples)
sigma_mle = np.std(samples)

print(f"\nMLE估计均值μ: {mu_mle:.4f}（真实值5）")
print(f"MLE估计标准差σ: {sigma_mle:.4f}（真实值2）")

# 可视化拟合效果
x = np.linspace(-2, 12, 100)
plt.figure(figsize=(8, 4))
plt.hist(samples, bins=30, density=True, edgecolor='black',alpha=0.6, label='样本分布')
#density=True：归一化。让直方图所有柱子的面积之和为 1，
#这样纵轴表示的是概率密度，而不是频数。只有这样才能和概率密度曲线直接比较
plt.plot(x, norm.pdf(x, mu_mle, sigma_mle), 'r-', label='MLE拟合正态分布')
#norm.pdf(x, mu_mle, sigma_mle)：计算正态分布的概率密度函数值。
#norm 来自 scipy.stats，表示正态分布。
#pdf 是概率密度函数（Probability Density Function）。
#mu_mle 和 sigma_mle 是用最大似然估计得到的均值和标准差。
#对于每个 x，返回对应的密度值。
#'r-'：指定线条样式为红色实线（r 表示 red，- 表示实线）。
plt.title('最大似然估计（正态分布）')
plt.legend()

# ===================== 4. 交叉熵（Cross Entropy） =====================
# 交叉熵是分类问题最常用的损失函数，衡量两个概率分布的差异
def cross_entropy(y_true, y_pred):
    #y_true：真实标签，取值0或1
    #y_pred：模型预测的概率，取值在0到1之间
    """二分类交叉熵：-sum(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))"""
    eps = 1e-7  # 防止log(0)
    y_pred = np.clip(y_pred, eps, 1-eps)
    #np.clip(数组, 最小值, 最大值)
    #小于最小值的元素 → 变成最小值；
    #大于最大值的元素 → 变成最大值；
    #在范围内的元素 → 保持不变。
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#在机器学习中，log默认指的是自然对数，也就是以e为底的对数，通常记作ln
# 真实标签
y_true = np.array([1, 0, 1, 0])
# 预测概率（模型输出）
y_pred_good = np.array([0.9, 0.1, 0.8, 0.2])  # 预测准确
y_pred_bad = np.array([0.1, 0.9, 0.3, 0.7])   # 预测错误

print(f"\n预测准确时的交叉熵:", cross_entropy(y_true, y_pred_good))
print(f"预测错误时的交叉熵:", cross_entropy(y_true, y_pred_bad))
print("结论：预测越准确，交叉熵越小；分类模型的目标就是最小化交叉熵")
plt.show()
