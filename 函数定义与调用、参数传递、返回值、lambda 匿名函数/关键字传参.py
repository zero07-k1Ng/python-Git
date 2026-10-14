def birthday(name,age):
    print(f'祝{name}{age}岁生日快乐')
birthday(age=23,name='zwt')#关键字
birthday('lzy',age=20)#如果同时使用位置和关键字传参，要保证：
#位置在前，关键字在后
