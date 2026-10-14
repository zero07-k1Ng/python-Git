import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成带噪声的正弦数据
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.2, X.shape)
#np.random.normal(0, 0.2, X.shape)：生成均值为 0、标准差为 0.2 的正态分布噪声，形状和 X 一样

# 不同多项式次数
degrees = [1, 4, 15]  # 1次=欠拟合，4次=合适，15次=过拟合

plt.figure(figsize=(15, 4))
X_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

for i, degree in enumerate(degrees):
    # 多项式特征转换
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)#先学习如何生成特征，然后转换 X
    X_test_poly = poly.transform(X_test)
    # 训练
    model = LinearRegression()
    model.fit(X_poly, y)

    # 预测
    y_pred = model.predict(X_poly)
    y_test_pred = model.predict(X_test_poly)
    train_mse = mean_squared_error(y, y_pred)

    plt.subplot(1, 3, i + 1)#plt.subplot(行数, 列数, 位置)：在一个画布上创建多个子图
    #1, 3：1 行 3 列，也就是并排放三张图
    #i + 1：当前子图放在第几个位置
    #循环中 i 从 0 开始：i=0 → 位置 1；i=1 → 位置 2；i=2 → 位置 3。
    #所以三个 degree 会分别画在左、中、右三张图上。
    plt.scatter(X, y, label='训练数据')
    plt.plot(X_test, y_test_pred, 'r-', label=f'{degree}次多项式')#模型预测曲线
    plt.plot(X_test, np.sin(X_test), 'g--', label='真实函数')#真实函数曲线
    plt.title(f'{degree}次多项式\n训练MSE={train_mse:.4f}')
    plt.legend()
    plt.ylim(-1.5, 1.5)#固定 y 轴范围

plt.tight_layout()#自动调整子图之间的间距，避免标题、坐标轴标签互相重叠。
plt.show()
print("结论：")
print("1次多项式：直线，无法拟合曲线 → 欠拟合")
print("4次多项式：基本拟合真实规律 → 拟合良好")
print("15次多项式：完美贴合训练点，但波动剧烈 → 过拟合")
