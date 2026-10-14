'''answer=input('今天要上学吗(y/n)：')
while answer=='y':
    print('好好学习')

    answer=input('今天要上学吗(y/n)：')
'''
s=0
i=0
while i<101:
    s+=i
    i+=1
else:#若while正常执行完，才可执行elsea,否则就不会执行else
    print(s)