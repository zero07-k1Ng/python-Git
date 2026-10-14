def copy(ylj,xlj):
    file1=open(ylj,'rb')
    file2=open(xlj,'wb')
    s=file1.read()
    file2.write(s)
    file2.close()#先打开的后关，后打开的先开
    file1.close()
if __name__ == '__main__':
    ylj='./mimi.jpg'#'.'代表当前目录
    xlj='../列表、字典、元组、集合/mimi.jpg'#'..'表示回退到上级目录
    copy(ylj,xlj)