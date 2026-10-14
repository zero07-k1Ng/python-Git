'''
x=eval(input('请输入你的3位彩票号码：'))
if x!=413:
    print('未中奖')
else:
    print('中奖了')
'''
x=eval(input('请输入你的3位彩票号码：'))
print('恭喜中奖' if x==413 else '未中奖')