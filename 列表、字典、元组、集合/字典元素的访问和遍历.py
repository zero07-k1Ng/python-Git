t={10:'zhang',20:'yuan',30:'ying'}
print(t[10])#1.使用t[key]来访问字典中键值对的值
print(t.get(10))#2.使用t.get(key)来访问字典中键值对的值
for i in t.items():
    print(i)#以元组形式输出
for key,value in t.items():
    print(key,value)#分别获取key和value