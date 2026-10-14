try:
    num1=int(input('请输入第一个数：'))
    num2=int(input('请输入第二个数：'))
    print('结果是：',num1/num2)
except ZeroDivisionError:
    print('除数为0')
except ValueError:
    print('除数为字符')
except BaseException:
    print('未知异常')