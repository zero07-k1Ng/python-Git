'''
def get_sum(num):#num为形式参数
    s=0
    for i in range(1,num+1):
        s+=i
    print(f'1到{num}的累加和为：{s}')

get_sum(10)#10为实际参数
get_sum(50)#50为实际参数
get_sum(100)
'''
def get_sum(num=50):#num为形式参数,当给num赋值时，为默认值参数，调用时可以不用再赋值
    s=0
    for i in range(1,num+1):
        s+=i
    print(f'1到{num}的累加和为：{s}')

get_sum()#如此
#如果同时用位置参数和默认值参数，在前面的一定是位置参数，后面可以是关键字参数也可以是默认值参数