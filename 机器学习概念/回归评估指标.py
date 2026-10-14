import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 真实值与预测值
y_true = np.array([3, 5, 7, 9, 11])
y_pred = np.array([2.8, 5.2, 6.9, 9.3, 10.8])

mse = mean_squared_error(y_true, y_pred)   # 均方误差
mae = mean_absolute_error(y_true, y_pred)  # 平均绝对误差：平均每个预测错了多少
rmse = np.sqrt(mse)                        # 均方根误差：MSE 开根号
r2 = r2_score(y_true, y_pred)              # R²决定系数

print("回归评估指标：")
print(f"MSE 均方误差: {mse:.4f}")
print(f"RMSE 均方根误差: {rmse:.4f}")
print(f"MAE 平均绝对误差: {mae:.4f}")
print(f"R² 决定系数: {r2:.4f}")
print("R²越接近1，模型解释能力（预测）越强")
