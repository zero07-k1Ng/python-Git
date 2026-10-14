a=100
def num(x,y):
    return a+x+y
print(num(10,20))
print(a)

def num1(x,y):
    a=200#此时的a为局部变量，当局部和全局变量名称相同时，局部变量优先级更高
    return a+x+y
print(num1(10,20))
print(a)

def num1(x,y):
    global k#在函数体内部可用global定义全局变量
    k=10#声明和赋值必须分开执行
    return k+x+y
print(num1(10,20))
print(k)