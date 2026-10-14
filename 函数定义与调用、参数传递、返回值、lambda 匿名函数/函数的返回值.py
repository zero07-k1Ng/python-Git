def fun(a,b):
    print(a+b)
fun(1,2)
print(fun(1,2))#此函数是无返回值函数，所以print的结果为None

def fun1(a,b):
    s=a+b
    return s #将s返回给函数调用处
x=fun1(10,20)#第一种：返回值存储到变量中
print(x)
x=fun1(fun1(10,20),3)#第二种：返回值参与计算，先计算最里面的函数，返回结果为后，再参与外面的函数计算
print(x)

def sum(num):
    s=0
    js=0
    os=0
    for i in range(1,num+1):
        if i%2==0:
            os=os+i
        else:
            js=js+i
        s=s+i
    return s,js,os#return后有3个值，返回3个值,多个返回值的话，结果为元组类型
x=sum(10)
print(type(x))
print(sum(10))
#解包赋值
a,b,c=sum(10)
print(a)
print(b)
print(c)