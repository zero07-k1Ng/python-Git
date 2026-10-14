import pandas as pd
df = pd.read_csv('greenhouse.csv')

#针对全部列
print(df.mean(numeric_only=True))#df.mean(numeric_only=True)：找出所有是数值列的平均值
print(df.sum(numeric_only=True))#df.sum(numeric_only=True):所有数值列求和
print(df.min(numeric_only=True))#df.min(numeric_only=True):所有数值列最小值
print(df.max(numeric_only=True))#df.max(numeric_only=True):所有数值列最大值
print(df.count())#统计每列中的数值量

#针对单个列
print(df['温度'].mean(numeric_only=True))#df.mean(numeric_only=True)：找出所有是数值列的平均值
print(df['湿度'].sum(numeric_only=True))#df.sum(numeric_only=True):所有数值列求和
print(df['光照强度'].min(numeric_only=True))#df.min(numeric_only=True):所有数值列最小值
print(df.max(numeric_only=True))#df.max(numeric_only=True):所有数值列最大值
print(df.count())#统计每列中的数值量

a=df.groupby('温度')#通过'温度'列进行分组df.groupby()
print(a['湿度'].mean())#结果为：在某确定温度下，该温度下所有湿度的平均值
print(a['湿度'].sum())#结果为：在某确定温度下，该温度下所有湿度的和
print(a['湿度'].count())#结果为：在某确定温度下，该温度下所有湿度的个数


