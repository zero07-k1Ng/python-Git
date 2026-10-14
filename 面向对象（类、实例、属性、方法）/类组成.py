class Student():
    school='安徽教育'#类属性
    #初始化方法，方法名必须是：__init__(self)
    def __init__(self,xm,age):
        self.name=xm#实例属性self.name
        self.age=age
    #实例方法
    def show(self):
       print(f'我叫{self.name},今年{self.age}岁了')#实例属性可以在整个类当中去使用，局部变量依然不行
    # 静态方法
    @staticmethod
    def sm():#静态方法中的形式参数部分，不自带self
        print('静态方法中，不能调用实例属性和实例方法')
    # 类方法
    @classmethod
    def cm(cls):#cls是class的简写
        print('这是类方法，不能调用实例属性和实例方法')
#创建类的对象
stu=Student('zwt',21)#__init__中的形参对应的就是这个类的对象里的参数，self无需理会，系统自带

#每个东西的调用
print(stu.name,stu.age)#实例属性的调用：对象名+"."
stu.show()#实例方法的调用：对象名+"."+方法名
print( )#类属性的调用：类名+"."
Student.cm()#类方法的调用：类名+"."+方法名
Student.sm()#静态方法的调用：类名+"."+方法名



