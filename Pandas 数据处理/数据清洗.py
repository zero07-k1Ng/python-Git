import numpy as np
import pandas as pd

# 让 Pandas 正确识别中英文混合时的显示宽度
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

df = pd.read_csv('greenhouse.csv')
#3. 缺失值处理
'''
df=df.drop(columns=['编号','时间'])#删除不想要的列
df=df.dropna(subset=['传感器状态'])#删除'传感器状态'这一列中所有没有值的行,即存在缺失值、空值、NaN
df=df.fillna({'传感器状态':'none'})#将'传感器状态'这一列中所有是(空值/NaN(非数值))的行，替换为'none'
df['传感器状态']=df['传感器状态'].replace({'Calibration_Warning': '校准警报'})#将'传感器状态'这一列中是'Calibration_Warning'的替换为：'校准警报'
#df['传感器状态']=df['传感器状态'].astype(bool)       :将'传感器状态'这一列中的值改为'布尔类型'
print(df.to_string())
#删除重复值
df=df.drop_duplicates()
print(df.to_string())

df.loc[df['温度']<=21,['温度','湿度']]=np.nan#逗号前面的部分，用来筛选行。告诉电脑：“我要找哪些行？”,逗号后面的部分，用来指定列。告诉电脑：“找到这些行后，我要修改哪一列的数据？”赋值操作。把定位到的“特定行、特定列”的格子，填入新的内容。
#同时修改多列的值的时候，要给两个列加一个[]
df['温度']=df['温度'].interpolate()#线性插值(传感器缺了十几分钟的数据，用前后时刻的值线性补全)
print(df.to_string())
'''
#4. 异常值处理
# 方法：3σ法（统计方法，适合正态分布的传感器噪声）
# 均值±3倍标准差之外的值认为是异常值
m=df['温度'].mean()
s=df['温度'].std()
df.loc[np.abs(df['温度']-m)>3*s,'温度']=np.nan
df['温度']=df['温度'].interpolate()
print(df.to_string())



