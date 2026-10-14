import admin.my_admin as a  #导入时为：包名.模块名
a.fun()
from admin import my_admin as b
b.fun()
from admin.my_admin import fun
fun()