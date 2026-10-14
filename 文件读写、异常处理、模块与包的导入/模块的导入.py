import my_info
print(my_info.name)#调用模块中的函数或者变量时，要采用：模块名+'.'的形式调用
my_info.fun()
import my_info as a#可以把原模块名简化命名
print(a.name)
a.fun()

#通配符
from my_info import *   #这样即可以调用这个模块里的所有东西
print(name)#此时再调用就不需要另加模块名了
fun()