class Student():
    school='安徽教育'#类属性
    #初始化方法，方法名必须是：__init__(self)
    def __init__(self,xm,age):
        self.name=xm#实例属性self.name
        self.age=age
    #实例方法
    def show(self):
       print(f'我叫{self.name},今年{self.age}岁了')#实例属性可以在整个类当中去使用，局部变量依然不行
stu1=Student('张文韬',22)
stu2=Student('张化儒',43)
stu3=Student('王敏',41)
stu4=Student('张芷诺',12)
print(type(stu1))
print(type(stu2))
print(type(stu3))
print(type(stu4))
Student.school='dick'
#将学生对象储存到列表中
lst=[stu1,stu2,stu3,stu4]
for i in lst:
    i.show()#对象名+"."调用

