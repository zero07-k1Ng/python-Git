def my_write():
    file=open('a.txt','w',encoding='utf-8')#打开（创建）文件
    file.write('我是大帅哥')#操作文件（写）
    file.close()#关闭文件
def my_read():
    file=open('a.txt','r',encoding='utf-8')
    s=file.read()#操作文件（读）
    print(s)
    print(type(s))
    file.close()

if __name__ == '__main__':
    #my_write()
    my_read()