s=['hello','world',98]
s.append('lll') #在列表最后一个元素后，增加一个元素
print(s)
s.insert(1,3)#insert(index,x)在指定的index位置上插入元素x
print(s)
#s.clear()#清空列表
s.remove('hello')
print(s)
s.pop(1)#将索引为1的元素先取出，再删除
print(s)
s.reverse()#将原列表逆序,无返回值
print(s)

