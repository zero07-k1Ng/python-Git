a=10
b=3
z=a/b
print(z)
#float转成int类型后，只保留整数，无四舍五入部分
print(int(3.14))
print(int(3.9))
print(int(-3.14))
print(float('3.14'))
print(int('10')+int('20'))
#int转成float类型后，补0
print(float(3))
#print(float('3.1a')),3.1a这个字符串在浮点数中并不存在，所以是错误语句

#下面3个进制转换后，均为字符串类型
print('十进制转十六进制:',hex(264))
print('十进制转八进制:',oct(13))
print('十进制转二进制:',bin(7))
