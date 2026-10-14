s='helloworld'
if 'l' in s:
    print('l在字符串s中')
else:
    print('不在')
print(len(s),'len()')
print(max(s),'max()')#序列s中最大的元素（这里是利用ASCII码的大小进行比较的）
print(min(s),'min()')#s中最小的元素
print('o在s中第一次出现的序号:',s.index('o'))
print('o在s中出现的次数:',s.count('o'))