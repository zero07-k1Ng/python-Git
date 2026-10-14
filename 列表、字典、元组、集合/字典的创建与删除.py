d={10:'dick',20:'dong', 30:'pig'}#！！字典中的键，必须是不可变序列
print(d)

s1=[10,20,30,40]
s2=['a','b','c','d']
z=zip(s1,s2)#利用zip函数来创建字典
print(z)#运行结果为：<zip object at 0x00000195FD7BFD00>
#因为zip() 是映射函数，所以只能看到字典z的映射位置，并不能直接看到字典中的内容
d=dict(z)#通过dict()内置函数，将z转换成字典类型，才能看到里面的内容
print(d)

#使用参数创建字典
d=dict(cat=10,pig=20)
print(d)

t=(10,20,30)
d={t:'cat'}#元组可以作为字典的键，元组是不可变数据类型
print(d)
t=[10,20,30]
d={t:10}
#print(d)#TypeError: cannot use 'list' as a dict key (unhashable type: 'list')
#列表不可以作为字典中的键，因为列表是可变数据类型