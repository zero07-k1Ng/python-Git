s=eval(input('请输入你的分数：'))
if s<0 or s>100:
    print('输入有误，你个代笔')
else:
    if s>0 and s<60:
        print('未及格')
    elif s>60 and s<90:
        print('良好')
    else:
        print('优秀')