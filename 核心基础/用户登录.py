user=input('请输入用户名：')
password=input('请输入密码：')
i=1
while user!='zwt' or password!='413600':
    print('第', i, '次输入错误')
    i+=1
    if i<=3:
        print('请重新输入')
        user=input('请输入用户名：')
        password=input('请输入密码：')
    else:
        break
if i>3:
    print('今日次数已用尽，请明日再来')
else:
    print('登录成功')