import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('greenhouse.csv')
df['时间'] = pd.to_datetime(df['时间'])
df=df.set_index('时间')# 必须先把时间设为索引
df['小时'] = df.index.hour
hour=df.resample('10min').mean(numeric_only=True)

#1.折线图plt.plot ()：画时序变化
plt.figure(figsize=(8,4))#初始化一张空白画布，设置它的宽度为 8 英寸，高度为 4 英寸。这个比例（2:1）常用于学术论文中的单栏宽图。
# 画温度曲线，红色，线宽2，加标签
plt.plot(hour.index, hour['温度'], color='tab:red', linewidth=2, label='温度')
# 同一张图画湿度曲线，蓝色，虚线
plt.plot(hour.index, hour['湿度'], color='tab:blue', linewidth=2, linestyle='--', label='湿度')
plt.xlabel('小时',fontsize=12)#x轴标签+字号
plt.ylabel('数值',fontsize=12)
plt.title('温室温湿度变化曲线', fontsize=14)#设置图表标题
plt.legend(fontsize=10)#显示图例，用于区分图中的多条线或不同的数据点
#标准式：plt.legend(fontsize=None, ncol=1, title=None, frameon=True),
#ncol: 图例的列数（如果图例项很多，可以设为2或3排成多列）
#frameon: 是否显示图例的边框（True/False）
plt.grid(alpha=0.3)#在背景中画出参考网格线,alpha=0.3表示透明度为30%
plt.savefig('温度曲线.png',dpi=300,bbox_inches='tight')
#bbox_inches='tight':它会裁掉图片边缘多余的空白，并保证如果你设置的坐标轴标签太长，保存图片时不会被截断在画面外面。
plt.show()#plt.show() 必须放在 plt.savefig() 的后面

#2.散点图plt.scatter ()：看两个变量的相关性
plt.figure(figsize=(6,4))
# alpha=0.5设置点透明度，避免点重叠看不清
plt.scatter(df['温度'], df['湿度'], alpha=0.5, color='tab:green', s=10)
#s=10: 设定数据点的大小（size）
plt.xlabel('温度(℃)')
plt.ylabel('湿度(%)')
plt.title('温度与湿度相关性散点图')
plt.savefig('温湿度散点.png', dpi=300, bbox_inches='tight')
plt.show()

#3. 直方图 plt.hist ()：看数据分布
plt.figure(figsize=(6,4))
# bins=20是分20个柱子，edgecolor加黑边更清晰
plt.hist(df['温度'], bins=20, edgecolor='black', alpha=0.7, color='tab:orange')
plt.xlabel('温度(℃)')
plt.ylabel('频次')
plt.title('温度值分布直方图')
plt.savefig('温度分布.png', dpi=300, bbox_inches='tight')
plt.show()

#4.箱线图sns.boxplot ()：看数据离散程度、识别异常值
plt.figure(figsize=(8,4))
# x是分组（小时），y是温度，data是DataFrame
sns.boxplot(x='小时', y='温度', data=hour, color='tab:blue')
plt.xlabel('小时')
plt.ylabel('温度(℃)')
plt.title('不同小时温度分布箱线图')
plt.savefig('温度箱线图.png', dpi=300, bbox_inches='tight')
plt.show()
#箱线图上的每一个图形（包含蓝色的箱子和上下的黑线）都代表了5个关键的统计数值，从上到下分别是：
#上边缘（最上面的横线）： 代表数据的最大值（不包含异常值）。
#上四分位数（箱子的顶边，Q3）： 代表有 75% 的数据都小于这个值。
#中位数（箱子中间的黑线）： 代表这组数据最中间的那个值（刚好 50% 的数据大于它，50% 小于它）。这是看平均水平最核心的指标。
#下四分位数（箱子的底边，Q1）： 代表有 25% 的数据小于这个值。
#下边缘（最下面的横线）： 代表数据的最小值（不包含异常值）。
#注：蓝色的“箱子”本身包含了最中间 50% 的数据。箱子越长，说明数据波动越大；箱子越扁，说明数据越集中。)

#5.热力图sns.heatmap ()：画相关性矩阵、多变量对比
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
#corr_matrix: 这是传入的数据，通常是一个包含相关系数的矩阵
#annot=True显示相关系数数值
#cmap='coolwarm': 设定颜色的映射方案,coolwarm意味着低数值用冷色调（蓝色）表示，高数值用暖色调（红色）表示
#vmin=-1, vmax=1: 设定颜色条（图例右侧的色带）的最大值和最小值。因为相关系数的取值范围严格限定在 -1 到 1 之间
#square=True: 强制将每个单元格绘制成正方形
plt.title('环境变量相关性热力图')
plt.savefig('相关性热力图.png', dpi=300, bbox_inches='tight')
plt.show()
