t=('hello',30,'dick')
print(t[1])#元组索引查找元素
t1=t[0:3:2]#元组切片
print(t1)
for i in t:
    print(i)
for i in range(len(t)):
    print(i,t[i])
for index,item in enumerate(t,start=1):#start=1是用来修改起始元素的序号值，这里修改成1
    print(index,item)
