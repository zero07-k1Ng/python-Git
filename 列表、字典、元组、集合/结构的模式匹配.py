date=eval(input('请输入数据类型：'))
match date:
    case {'name':'zwt','age':23}:
        print('字典')
    case [10,20,30]:
        print('列表')
    case (10,20,40):
        print('元组')
    case _:#这个语句类似于if-else中的else部分
        print('无')
