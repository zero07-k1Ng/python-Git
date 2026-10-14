import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 导数（数值计算） =====================
# 函数 f(x) = x² + 3x + 2，导数 f’(x) = 2x + 3
def f(x):
    return x ** 2 + 3 * x + 2

def numerical_derivative(func, x, eps=1e-5):
    """数值微分：用极限近似计算导数"""
    return (func(x + eps) - func(x - eps)) / (2 * eps)

x = 2
print(f"x={x}处的数值导数:", numerical_derivative(f, x))
print(f"x={x}处的解析导数:", 2 * x + 3)

# ===================== 2. 偏导数 =====================
# 二元函数 f(x,y) = x² + y² + xy
def f_xy(x, y):
    return x**2 + y**2 + x*y

def partial_derivative_x(func, x, y, eps=1e-5):
    """对x求偏导：y固定，x变化"""
    return (func(x+eps, y) - func(x-eps, y)) / (2*eps)

def partial_derivative_y(func, x, y, eps=1e-5):
    """对y求偏导：x固定，y变化"""
    return (func(x, y+eps) - func(x, y-eps)) / (2*eps)

x0, y0 = 1, 2
print(f"\n在({x0},{y0})处对x的偏导:", partial_derivative_x(f_xy, x0, y0))
print(f"解析解 ∂f/∂x = 2x + y =", 2*x0 + y0)
print(f"在({x0},{y0})处对y的偏导:", partial_derivative_y(f_xy, x0, y0))
print(f"解析解 ∂f/∂y = 2y + x =", 2*y0 + x0)

# ===================== 3. 链式法则 =====================
# 复合函数 f(g(x))，导数 = f’(g(x)) * g’(x)
# 例：f(u) = u², u = g(x) = 3x + 1
def g(x):
    return 3 * x + 1

def f_u(u):
    return u ** 2

def composite_f(x):
    return f_u(g(x))

# 数值求导
x = 1
comp_deriv = numerical_derivative(composite_f, x)
# 链式法则解析解：2*(3x+1)*3
chain_deriv = 2 * g(x) * 3

print(f"\n复合函数数值导数:", comp_deriv)
print(f"链式法则解析导数:", chain_deriv)

# ===================== 4. 梯度下降原理演示 =====================
# 目标：找到 f(x) = x² 的最小值
def loss_func(x):
    return x ** 2

def gradient(x):
    """梯度（导数）：2x"""
    return 2 * x

# 梯度下降参数
x = 10.0       # 初始值
lr = 0.1       # 学习率
epochs = 50    # 迭代次数
history = []   # 记录x的变化

for i in range(epochs):
    grad = gradient(x)
    x = x - lr * grad  # 核心公式：沿梯度反方向更新
    history.append(x)#把x作为一个元素追加到history的末尾
    print(history)

print(f"\n梯度下降最终x值:", x)
print(f"理论最小值点: 0")

# 可视化
x_plot = np.linspace(-10, 10, 100)
y_plot = loss_func(x_plot)

plt.figure(figsize=(8, 4))
plt.plot(x_plot, y_plot, label='损失函数 f(x)=x²')
plt.scatter(history, [loss_func(x) for x in history], color='red', label='梯度下降过程')
plt.xlabel('x')
plt.ylabel('loss')
plt.title('梯度下降过程可视化')
plt.legend()
plt.grid(True)
plt.show()
