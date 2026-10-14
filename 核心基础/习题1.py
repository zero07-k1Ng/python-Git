x=input('请输入一个四位数：')
print(eval(x)//1000)#千位
print(int((eval(x)%1000)/100))#百位
print(int((eval(x)%100)/10))#十位
print(eval(x)%10)#个位