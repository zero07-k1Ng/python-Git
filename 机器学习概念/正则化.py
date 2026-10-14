import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据
np.random.seed(42)
X = np.linspace(0, 2*np.pi, 20).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.2, X.shape)

# 15次多项式（容易过拟合）
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X)
X_test = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
X_test_poly = poly.transform(X_test)

# 无正则化
model_plain = LinearRegression()
model_plain.fit(X_poly, y)
y_plain = model_plain.predict(X_test_poly)

# L2正则化（岭回归）#Ridge在原来的平方误差上增加了一个惩罚项:罚系数平方和
model_ridge = Ridge(alpha=1.0)#alpha=1.0 ：惩罚强度。越大，惩罚越强，模型越简单
model_ridge.fit(X_poly, y)
y_ridge = model_ridge.predict(X_test_poly)
#让所有系数整体变小，曲线更平滑，不容易剧烈震荡。
# L1正则化（Lasso）#Lasso的惩罚项是绝对值
model_lasso = Lasso(alpha=0.01, max_iter=10000)#max_iter=10000：Lasso使用迭代算法求解，增加最大迭代次数，避免不收敛警告
model_lasso.fit(X_poly, y)
y_lasso = model_lasso.predict(X_test_poly)
'''
Ridge 的 alpha=1.0 和 Lasso 的 alpha=0.01 不能直接比较谁更强
因为惩罚形式不同，而且这里的多项式特征没有标准化，尺度差异很大
'''
# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='训练数据')
plt.plot(X_test, np.sin(X_test), 'g--', label='真实函数')
plt.plot(X_test, y_plain, 'r-', label='无正则化（过拟合）')
plt.plot(X_test, y_ridge, 'b-', label='L2正则化 Ridge')
plt.plot(X_test, y_lasso, 'orange', label='L1正则化 Lasso')
plt.ylim(-1.5, 1.5)
plt.title('正则化缓解过拟合效果')
plt.legend()
plt.grid(True)
plt.show()

print("参数数量对比（L1会让很多参数变为0）：")
print(f"无正则化非零参数数: {np.sum(model_plain.coef_ != 0)}")
print(f"Ridge非零参数数: {np.sum(model_ridge.coef_ != 0)}")
print(f"Lasso非零参数数: {np.sum(np.abs(model_lasso.coef_) > 1e-5)}")
