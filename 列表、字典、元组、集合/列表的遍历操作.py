lst=['hello','world',98]
for item in lst:
    print(item)
print('-'*50)
for i in range(len(lst)):
    print(lst[i])
#enumerate()函数
for index,item in enumerate(lst):
    print(index,item)#index是序号不是索引，可以手动修改
for index,item in enumerate(lst,start=1):
    print(index,item)