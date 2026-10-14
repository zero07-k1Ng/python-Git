def fun(*name):#个数可变的（位置）参数：在形参前加*，此时参数变成元组类型，参数可以不止一个了
    for i in name:
        print(i)
fun(1,2,3,4)
fun([1,2,3,4])#一个列表被当成一个元素，所以整体被输出出来
fun(*[1,2,3,4,])#调用时，参数前加一个*，可以将列表解包，把列表中的每个元素输出出来

def fun1(**name):#个数可变的（关键字）参数：在形参前加**，此时参数变成字典类型，参数可以为任意个
    print(type(name))
    for k,v in name.items():#name.items()是字典遍历的写法公式
        print(k,v)
fun1(name='zwt',age=23,height=183)
d={'name':'zwt','age':21,'height':183}
print(**d)#如果传递的参数是一个字典，那么需要在这个字典前加**，来进行解包才可正常运行