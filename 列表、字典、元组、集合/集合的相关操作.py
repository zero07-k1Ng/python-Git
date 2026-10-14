s={10,20,30}
s.add(40)#添加元素x
print(s)
s.remove(20)#删除元素x
print(s)
'''
s.clear()#清除集合s
print(s)
'''
for i in s:
    print(i)
for index,value in enumerate(s,start=1):#在这里index是序号，不是索引
    print(index,'---->',value)
s={i for i in range(1,10) if i%2==0}#集合生成式，if的结果如果是True，则执行if前面的语句
print(s)

