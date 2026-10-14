s='3.14+3'
print(s,type(s))
x=eval(s)#用于去掉字符串最外侧的引号，并执行去掉引号后的部分
print(x,type(x))
age=eval(input('我的年龄：'))
print(age,type(age))
height=eval(input('我的身高：'))
print(height,type(height))
hello='你好世界'
print(eval('hello'))
#print(eval('你好世界'))是错误的；原因：name '你好世界' is not defined