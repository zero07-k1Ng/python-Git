t=(i for i in range(10))
print(t)#此时t为一个生成器对象
'''
t=tuple(t)
print(t)
for i in t :
    print(i)
'''
print(t.__next__())
print(t.__next__())
print(t.__next__())
t1=tuple(t)
print(t1)#因为使用__next__()函数将生成器中的元素取出来一部分了，所以再输出t时则会缺少被取出来的那部分