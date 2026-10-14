import pandas as pd
df=pd.read_csv('greenhouse.csv')
#df.to_csv('清洗后的数据.csv', index=False)  :将表导出并保存为名为'清洗后的数据.csv'，
#Pandas 默认会把行号（0, 1, 2...）作为第一列写入文件；设为 False 可以避免生成多余的无表头序号列。
sx=df[(df['湿度']>=45) & (df['温度']>=22)]#筛选多列标签，保留符合条件的行
#在组合多列条件时，每个独立的条件都必须用圆括号()包裹起来。
print(sx)

