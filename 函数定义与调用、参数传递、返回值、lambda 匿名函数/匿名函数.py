#当一个函数只有一句代码且只有一个返回值时，才可以使用
s=lambda a,b:a+b#s是一个函数
print(type(s))
print(s(10,20))

#列表取值
lst=[1,2,3,4,5]
for i in range(len(lst)):
    s=lambda x:x[i]
    print(s(lst))
students_scores=[
    {'name':'zwt','score':97},
    {'name':'zzn','score':95},
    {'name':'zhr','score':91},
    {'name':'wm','score':60}
]
students_scores.sort(key=lambda x:x['score'],reverse=True)
print(students_scores)