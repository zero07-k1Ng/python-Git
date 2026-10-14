import random
d={i:random.randint(1,101) for i in range(4)}
print(d)

s1=[10,20,30,40]#作为键
s2=['a','b','c','d']#作为值
d={k:v for k,v in zip(s1,s2)}
print(d)