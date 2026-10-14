def copy(ylj,xlj):
    with open(ylj,'r',encoding='utf-8') as file1:#with语句的读与写格式
        with open(xlj,'w',encoding='utf-8') as file2:
            file2.write(file1.read())
if __name__ == '__main__':
    copy('./x.txt','../列表、字典、元组、集合/x.txt')
