'''
x=eval(input('请输入一个年份：'))
if (x%4==0 and x%100!=0) or x%400==0:
    print(x,"年为闰年")
else:
    print(x,'年不是闰年')
'''

'''
x=input('请选择功能：')
match x:
    case '1':
        print('当前余额：60元')
    case '2':
        print('当前剩余流量：10G')
    case '3':
        print('当前剩余通话时长：100分钟')
    case '0':
        print('您已退出查询窗口')
'''
import random
x=random.randint(1,100)
s=eval(input('请输入你的数字：'))
i=1
while i<=10:
    if s==x:
        print('你在第',i,'次猜中了')
        break
    elif s<x:
        print('猜小了，重新猜：')
        i+=1
        print('第',i,'次')
        s = eval(input('请输入你的数字：'))
    elif s>x:
        print('猜大了')
        i += 1
        print('第', i, '次')
        s = eval(input('请输入你的数字：'))
else:
    print('10次你都没猜中')
