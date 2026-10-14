a=100
b=50
print(a+b)#print()中，如果是数字，不需要加''，不加默认为字符串类型
print(a)
print('a+b')
print (a,b,a+b,'a*b')

fp=open('dick.txt', 'w')
print('河北正宗安徽板面',file=fp)
fp.close()

