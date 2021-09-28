# coding: UTF-8
"""
@FileName: components.py
@author：shi yao  <aiwei169@sina.com>
@date: 2021/01/16 16:21
@Software: PyCharm
python version: python 3.6.13
"""

"""
通用代码配置
"""

import sys
import os
import datetime
import argparse
import logging
import warnings
import functools
from xml.etree import ElementTree as ET
from pathlib import Path


class PrintTree:
    def __init__(self, root_dir: str):
        """用于打印目录树, 提供方便的目录层级查看功能

        Args:
            root_dir: 路径字符串
        """
        self._str = self._generate_tree(Path(root_dir)) if root_dir else None

    @staticmethod
    def _generate_tree(pathname: Path, max_depth=-1, __n=0) -> str:
        """
        Note:
            n 仅作为内部递归调用使用，不作为外部参数
            str 路径表示的入口，请用类。此函数简化，仅用 Path 做参数传递
        """
        if max_depth == 0:
            return ''

        result = ['    |' * __n, '----']
        if pathname.is_file():
            result.extend([pathname.name, '\n'])
        elif pathname.is_dir():
            result.extend([str(pathname.relative_to(pathname.parent)), '/\n'])
            for cp in pathname.iterdir():
                result.append(PrintTree._generate_tree(cp, max_depth - 1, __n + 1))
        return ''.join(result)

    def __str__(self):
        return self._str


def deprecated(func):
    """

    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    example:

        @deprecated
        def some_old_function(x, y):
            return x + y

    Note:
        https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def to_list(s: str, fmt=None) -> list:
    """ 字符串序列格式化函数（空格分离）

    >>> to_list('5 20 35  60  ', fmt=int)
    [5, 20, 35, 60]
    """
    fmt = str if fmt is None else fmt
    return [fmt(i.strip()) for i in s.split()]


class Logger(logging.Logger):
    """ 日志操作对象

    功能设计：
        1. 提供初始化功能，输入log文件路径，初始化
        2. 默认记录level是 INFO， debug模式的level是 DEBUG
        3. 提供更改 format 的功能

    Note:
        因为 logging 模块反对直接从Logger实例化（要用getLogger方法），但是为了使用Logger属性方法，这里继承了 Logger，
        后果是，loggging无法识别本Logger的存在，需要在Logger层面操作handler和filter
        好处是，若挂载到logging.root，则可以通过logging.info 等方法调用（默认调用root），保持logging层面调用很方便
        所以，对于多Logger对象，需要用 manager进行管理
        这里设计初衷是只有一个logger，所以不考虑 manager
    """
    _format = '%(filename)20s:%(lineno)-4d [%(levelname)-5s] >> %(message)s'
    _level = logging.INFO  # 用于handlers 在log层面是NOTSET
    _level_debug = logging.DEBUG

    def __init__(self, log_path=''):
        # 用 log_path 作为 name ,并直接增加 Handler
        super().__init__(name=log_path, level=logging.NOTSET)
        self.time0 = datetime.datetime.now()
        self._formatter = logging.Formatter(self._format)
        self.add(log_path)
        self.set_format("%(asctime)s %(filename)s: %(lineno)d >> %(message)s", '%Y.%m.%d %H:%M:%S')

    def add(self, log_path=None) -> 'Logger':
        """ 增加handler

        如果是None或者''等，则增加屏幕输出；否则增加 FileHandler
        """
        if log_path:
            dir_name = os.path.dirname(log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding='utf-8')
        else:
            handler = logging.StreamHandler()
        handler.setLevel(self._level)
        handler.setFormatter(self._formatter)
        self.addHandler(handler)
        return self

    def set_format(self, fmt: str, datefmt: str):
        self._format = fmt
        self._formatter = logging.Formatter(fmt, datefmt)
        for a_handler in self.handlers:
            a_handler.setFormatter(self._formatter)

    def to_debug(self):
        for a_handler in self.handlers:
            a_handler.setLevel(self._level_debug)

    def to_normal(self):
        for a_handler in self.handlers:
            a_handler.setLevel(self._level)

    def print_tag_start(self):
        self.info("*" * 20)
        self.info("start: {}".format(self.time0))

    def print_tag_close(self):
        self.info("spent time: {}".format(datetime.datetime.now() - self.time0))
        self.info("*" * 20 + "\n")


class raiseE(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class XML(object):
    """ 业务配置文件对象

    约定：
        xml文件保留 __log__ 字段，
        关于内置日志方法
            会直接替换（挂载到） logging.root， 通过 logging 或者 logging.root 访问
            可外部通过 logging.info 等方法，用于日志记录，
    提供 set_log_format 方法
        可修改日志记录的前缀
    提供 parse 方法，用于增加xml配置信息
        通过继承该类，覆写 parse 方法，进行文件的其他内容的自定义解析
    提供 with 方法，用于业务代码主函数，规范日志和流程;
        使用上下文管理器，而不是修饰器，是为了脚本中暴露主要变量，方便调试
    """

    def __init__(self, xml_path: str, debug_mode: bool):
        self._xml_path = xml_path
        self._debug_mode = debug_mode
        self._logger = None

        self._tree = ET.parse(self._xml_path)
        if self._tree.find('__log__') is not None:
            self._init_log(self._tree.find('__log__').text)
        self.parse()
        del self._tree

    def parse(self):
        """
        # 建议 override, 用 tag 作为 key 或 __attribute__
        """
        self.path = {i.tag: i.text for i in self._tree.find('path')}
        self.param = int(self._tree.find('param').text)

    def set_log_format(self, fmt: str):
        if self._logger is not None:
            self._logger.set_format(fmt)
        else:
            raise RuntimeError('没有定义log对象，设置格式无效')

    def _init_log(self, path_log: str):
        self._logger = Logger(datetime.datetime.now().strftime(path_log))
        if self._debug_mode:
            self._logger.add()  # 当开启测试，且路径存在(当前为FileHandler)时，增加 StreamHandler
            self._logger.to_debug()
        logging.root = self._logger  # 替换 logging.root

    @classmethod
    def to_list(cls, s: str, fmt=None) -> list:
        return to_list(s, fmt)

    def __str__(self):
        return '\n'.join('{:<16}:{}'.format(k, v) for k, v in self.__dict__.items())

    def __enter__(self):
        if self._logger is not None:
            self._logger.print_tag_start()
        else:
            raise RuntimeError('没有定义log对象，无法使用 with 方法')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            logging.exception(exc_val)
        self._logger.print_tag_close()
        return True  # 内部处理错误，所以 return True； 如果不处理错误，则返回 None or False


class Arg(argparse.ArgumentParser):
    """ 参数解析类

    约定：
        保留 -t --time, -d --debug, -xml --xml_path 的标识符

    >>> parser = Arg()
    >>> parser.print_help()
    usage: docrunner.py [-h] [-c] [-r] [-d] [-t [TIME [TIME ...]]] [-xml XML_PATH]
    <BLANKLINE>
    optional arguments:
      -h, --help            show this help message and exit
      -d, --debug           如果输入-d，则表示进行debug，输出logging，否则不输出
      -t [TIME [TIME ...]], --time [TIME [TIME ...]]
                            指定时刻或时段，如：201906110900，或：201906110900
                            201906111000，默认为当前北京时刻
      -xml XML_PATH, --xml_path XML_PATH
                            输入配置文件路径, 默认值为：config/config.xml
    """

    def __init__(self, description=''):
        super().__init__(description=description)
        self.add_argument('-d', '--debug', action='store_true', help='如果输入-d，则表示进行debug，输出logging，否则不输出')
        self.add_argument('-t', '--time', default=datetime.datetime.now().strftime('%Y%m%d%H%M'),
                          help='指定时刻或时段，如：201906110900，或：201906110900 201906111000，默认为当前北京时刻', nargs='*')
        self.add_argument('-xml', '--xml_path', default=os.path.join(sys.path[0], 'config', 'config.xml'),
                          help='输入配置文件路径, 默认值为：config/config.xml')

    def arg_parse(self, args_list=None):
        args = self.parse_args(args_list)
        return args