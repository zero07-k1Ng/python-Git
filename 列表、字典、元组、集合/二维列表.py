lst=[['城市','环比','同比'],['北京',101,102],['上海',90,103],['深圳',102,100],]
print(lst)
for row in lst:#行
    for item in row:#列
        print(item,end='\t')
    print()