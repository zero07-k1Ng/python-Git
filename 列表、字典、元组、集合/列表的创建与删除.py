s=['dick',98,100.5]
print(s)
s1=list('hello')
s2=list(range(1,11,2))#从1开始到11结束，不包含11，步长为2
print(s1)
print(s2)
print(s1+s2)
del s1#s1列表被删除，下一行的print(s1)将报错，因为s1此时已经不存在了
print(s1)