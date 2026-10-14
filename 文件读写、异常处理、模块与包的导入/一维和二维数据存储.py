lst=['张文韬','王敏','张芷诺','张化儒']
def my_write():
    #.csv文件的全称叫“逗号分隔值文件”（用逗号来区分不同的数据）
    with open('family.csv','w',) as file:#.csv是带逗号分隔的文件
        file.write(','.join(lst))#把列表lst里的每一个元素拿出来，
        #用前面的',（逗号）作为连接符，把它们连接在一起
def my_read():
    with open('family.csv','r',) as file1:
        s=file1.read()
        lst=s.split(',')#从头到尾检查字符串s，只要一看到','（逗号）
         #就把它切开，然后把切开的每一块放进一个新的列表里。
        print(lst)
def my_write_table():
    lst1=[
    ['名称','价格','数量'],
    ['篮球','50.8','10'],
    ['鼠标','20.9','5']
    ]
    with open('family1.csv','w') as file:
        for i in lst1:
            file.write(','.join(i))
            file.write('\n')
def my_read_table():
    data=[]
    with open('family1.csv','r') as file:
        lst=file.readlines()
        print(lst)
        print(len('\n'))#结果为“1”，所以'\n'本质上只占一个字符的位置
        for i in lst:
            new_lst=i[:len(i)-1].split(',')#所以是len(i)-1
            data.append(new_lst)
    print(data)



if __name__ == '__main__':
    #my_write()
    #my_read()
    my_write_table()
    my_read_table()

