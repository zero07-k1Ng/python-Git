import pandas as pd

date={'name':['zwt','wm','zzn'],#name和age均为列名，[]内的数据均为每一列的内容
      'age':[23,46,13]}
df=pd.DataFrame(date,index=['员工1','员工2','员工3'])
#print(df.loc['员工1'])------df.loc[]通过具体索引确定信息
#print(df.iloc[1])------df.iloc[]通过序列号来确定详细信息
df['job']=['cook','boss','waiter']#创建新列
#创建新行
new_row=pd.DataFrame([{'name':'zhr','age':47,'job':'driver'},
                   {'name':'zero','age':48,'job':'manger'}]
                     ,index=['员工4','员工5'])
df=pd.concat([df,new_row])#将新行和旧行合并起来
print(df)