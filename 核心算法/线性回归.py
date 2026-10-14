import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 准备数据 =====================
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 特征
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5  # 真实关系：y=3x+4 + 噪声

# ===================== 2. 手动实现：正规方程（闭式解） =====================
# 公式：θ = (X^T X)^{-1} X^T y
X_b = np.c_[np.ones((100, 1)), X]  # 添加偏置项x0=1
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("正规方程求得的参数：")
print(f"截距b = {theta_normal[0][0]:.2f}")
#.4f 表示保留 4 位小数
print(f"权重w = {theta_normal[1][0]:.2f}")

# ===================== 3. 手动实现：梯度下降法 =====================
def gradient_descent_linear(X, y, lr=0.1, epochs=1000):
    m = len(X)
    theta = np.random.randn(2, 1)  # 初始化参数
    for epoch in range(epochs):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)  # 损失函数MSE的梯度
        theta = theta - lr * gradients
    return theta

theta_gd = gradient_descent_linear(X_b, y, lr=0.1, epochs=1000)
print("\n梯度下降求得的参数：")
print(f"截距b = {theta_gd[0][0]:.2f}")
print(f"权重w = {theta_gd[1][0]:.2f}")

# ===================== 4. sklearn 实现 =====================
model = LinearRegression()#创建一个线性回归模型对象
model.fit(X, y)#训练模型，特征 X 和目标 y
#fit 内部使用的是最小二乘法（本质就是正规方程）
print("\nsklearn 线性回归参数：")
print(f"截距b = {model.intercept_[0]:.2f}")
print(f"权重w = {model.coef_[0][0]:.2f}")

# 预测
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"均方误差MSE: {mse:.2f}")

# 可视化
plt.figure(figsize=(8, 4))
plt.scatter(X, y, alpha=0.6, label='样本数据')
plt.plot(X, y_pred, 'r-', linewidth=2, label='拟合直线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归拟合效果')
plt.legend()
plt.grid(True)
plt.show()
