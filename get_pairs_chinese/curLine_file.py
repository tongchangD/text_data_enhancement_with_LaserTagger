# -*- coding: utf-8 -*-
import sys
def curLine():
    file_path = sys._getframe().f_back.f_code.co_filename  # 获取调用函数的路径
    file_name=file_path[file_path.rfind("/") + 1:] # 获取调用函数所在的文件名
    lineno=sys._getframe().f_back.f_lineno#当前行号
    str="[%s:%s] "%(file_name,lineno)
    return str