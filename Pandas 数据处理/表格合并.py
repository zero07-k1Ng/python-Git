import pandas as pd
'''
多个结构一样的表上下拼起来
（比如你 1 月、2 月、3 月各一个传感器 csv，拼成一个季度的总表）
'''
df1 = pd.read_csv('1月数据.csv')
df2 = pd.read_csv('2月数据.csv')
df3 = pd.read_csv('3月数据.csv')
df_all = pd.concat([df1, df2, df3], axis=0)#axis=0是上下拼

'''
两个表按共同的列（比如时间）拼起来
（比如把环境数据表和作物生长测量表按时间合并，分析环境对作物的影响）
'''
# df_env是环境数据（时间、温度、湿度），df_growth是生长数据（时间、株高、叶面积）
df_merge = pd.merge(df_env, df_growth, on='时间', how='inner')#按时间列匹配
#on='时间': 指定合并的基准列。系统会去左表和右表中寻找名为“时间”的列，并把时间相同的行拼凑在一起。
#how='inner': 指定合并的方式。inner 代表“内连接”（取交集）。如果某个时间点只有左表有，而右表没有，那么这行数据会被直接丢弃。

