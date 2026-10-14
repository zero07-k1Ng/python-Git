import pandas as pd
df = pd.read_csv('greenhouse.csv',index_col='时间')
# 1. 读csv文件（传感器导出的文件基本都是csv）
#df = pd.read_csv('greenhouse.csv')  # 括号里填你的文件路径和类型名
#print(df.to_string())#使用.to_string()函数，则会打印所有数据，否则只打印前5行和后5行
#按列进行选择
'''
print(df['温度'].to_string())
print('-'*50)
print(df[['湿度','温度']])#df[]只打印一列，多列需要使用df[[]]

#按行选择
print(df.loc[1])
df = pd.read_csv('greenhouse.csv',index_col='时间')#index_col='时间'，则是用“时间”作为索引
print(df.loc['2026-07-28 09:56:00'])
print(df.iloc[0:11:2,0:3])#第一个参数：从0到10，步长为2；第二个参数：只要前3列
'''
name=input('请输入一个时间戳：')
try:
    print(df.loc[name])
except KeyError:
    print(f'这个{name}不存在数据集中')
    name = input('请重新输入一个时间戳：')
    print(df.loc[name])






