# coding: UTF-8
"""
@Project: FY4A
@FileName: fy4a.py
@author: shi yao  <aiwei169@sina.com>
@create date: 2021/9/17 13:54
@Software: PyCharm
python version: python 3.6.13
"""


import struct
from typing import Tuple
import numpy as np
import utils
import xarray as xr
from numpy import ndarray
import scipy.interpolate as ip
import os
import components
import logging
import datetime as dt


class CFG(components.XML):
    """
    模型配置类
    """
    def __init__(self, args: list or str, screen_show_info: bool = True) -> None:
        # 针对自己的xml，覆写XML的parse方法
        self.now = dt.datetime.now()  # 获取当前时间
        self.screen_show_info = screen_show_info  # 是否展示基本信息
        self.args = args
        super().__init__(self.args.xml_path, self.args.debug)

    def parse(self):
        # 覆写argparse的parse方法
        self.path = {i.tag: i.text for i in self._tree.find('path')}
        self.save_path = {i.tag: i.text for i in self._tree.find('save_path')}
        self.location = {i.tag: i.text for i in self._tree.find('location')}
        self.param = {i.tag: i.text for i in self._tree.find('param')}
        if self.screen_show_info:
            logging.debug("除非特殊说明，否则debug中显示的时间、argparse的时间参数、生成的产品命名的时间均为北京时间！")
            logging.debug("完成配置文件读取：{}".format(self._xml_path))

    @staticmethod
    def raiseE(msg: str):
        raise components.raiseE(msg)


class FY4A(CFG):
    """
    读取风云四号共57种卫星产品，并提供插值到指定区域的功能
    除GIIRS外，向每个数据dataset对象中写入主产品数据块对应的lat矩阵，lon矩阵，flag矩阵，lat,lon,flag的大小与主产品的数据块大小一致
    flag为1代表数据可用，为0代表不可用
    没有对原始数据做任何的更改，没有做定标、定位和插值，但是提供了定标、定位以及插值的方法供用户调用
    """
    def __init__(self, args: list or str, screen_show_info: bool = True):
        super().__init__(args, screen_show_info)
                                                # 计算参数设置
        self.pi = 3.1415926535897932384626      # 圆周率
        self.ea = 6378.137                      # 地球的半长轴
        self.eb = 6356.7523                     # 地球的短半轴
        self.h = 42164                          # 地心到卫星质心的距离
        self.lambda_d = 104.7                   # 卫星星下点所在的经度
        self.l_max = {                          # 标称投影上的行号的最大值(从0开始)
            '500m': 21983,
            '1000m': 10991,
            '2000m': 5495,
            '4000m': 2747,
            '12000m': 916
        }
        self.COFF = {                           # 列偏移
            '500m': 10991.5,
            '1000m': 5495.5,
            '2000m': 2747.5,
            '4000m': 1373.5,
            '12000m': 457.5
        }
        self.CFAC = {                           # 列比例因子
            '500m': 81865099,
            '1000m': 40932549,
            '2000m': 20466274,
            '4000m': 10233137,
            '12000m': 3411045
        }
        self.c_max = self.l_max                 # 标称投影上的列号的最大值
        self.LOFF = self.COFF                   # 行偏移
        self.LFAC = self.CFAC                   # 行比例因子

                                                # 经纬度范围设置
        self.lat_max = 80.56672132              # 标称图上全圆盘经纬度有效范围
        self.lat_min = -80.56672132             # 标称图上全圆盘经纬度有效范围
        self.lon_max = 174.71662309             # 标称图上全圆盘经纬度有效范围
        self.lon_min = 24.11662309              # 标称图上全圆盘经纬度有效范围

        self.raw_path = {                       # raw源文件的位置
            '500m': self.path['raw_name_500m'],
            '1000m': self.path['raw_name_1000m'],
            '2000m': self.path['raw_name_2000m'],
            '4000m': self.path['raw_name_4000m'],
            '8000m': self.path['raw_name_8000m'],
            '16000m': self.path['raw_name_16000m']
        }

        self.product_name = {                   # 数据字段名与产品名不一致的情况
            'ACI': 'Channel0161',
            'CFR': 'CFR',
            'CIX': 'CTC',
            'LPW': 'LPW_HIGH',
            'QPE': 'Precipitation',
            'TBB': 'NOMChannel07',
            'TFP': 'estimated_TP',
            'AMV': 'col'
        }

    def get_info_from_file_path(self, file_path: str) -> dict:
        # 根据文件路径获取文件名，并根据文件名获取各个字段信息，集成为字典
        info = file_path.split(os.sep)[-1].replace('-', '').split('_')
        temp = info[-1].split('.')
        del info[-1]
        info.append(temp[0])
        info.append(temp[1])                        # 把文件名分割成字段列表

        if info[11] == '064KM':
            info[11] = '48KM'                       # 名字上写着64KM，实际上是48KM分辨率
        if 'K' in info[11]:
            info[11] = info[11].replace('K', '000')
        info[11] = str(int(info[11][:-1])) + 'm'    # 标准化分辨率参数

        info = {
            'Satellite_Name': info[0],              # 卫星名称
            'Sensor_Name': info[1],                 # 仪器名称
            'Observation_Mode': info[2],            # 观测模式
            'Area': info[3],                        # 数据区域类型
            'Sub_astral_Point': info[4],            # 星下点经纬度
            'Level': info[5],                       # 数据级别
            'Product_Name': info[6],                # 数据名称
            'Band': info[7],                        # 仪器通道名称
            'Project': info[8],                     # 投影方式
            'Start_Time': info[9],                  # 观测起始日期时间(UTC)
            'End_Time': info[10],                   # 观测结束日期时间(UTC)
            'Resolution': info[11],                 # 空间分辨率
            'Backup': info[12],                     # 备用字段
            'Format': info[13]                      # 数据格式
        }
        return info

    def fix_lat_lon(self, lat: ndarray, lon: ndarray, **kwargs) -> Tuple[ndarray, ndarray]:
        lat_temp = lat.copy()
        lon_temp = lon.copy()
        # 根据官方指定的经纬度范围做修正
        lat_temp[lat > self.lat_max] = np.nan
        lat_temp[lat < self.lat_min] = np.nan
        lon_temp[lon > self.lon_max] = np.nan
        lon_temp[lon < self.lon_min] = np.nan
        if kwargs.get('l') is not None and kwargs.get('c') is not None:
            lon_temp[kwargs['l'] == -1] = np.nan
            lon_temp[kwargs['c'] == -1] = np.nan
            lat_temp[kwargs['l'] == -1] = np.nan
            lat_temp[kwargs['c'] == -1] = np.nan
        return lat_temp, lon_temp

    def fix_l_c(self, l: ndarray, c: ndarray, res: str) -> Tuple[ndarray, ndarray]:
        l_temp = l.copy()
        c_temp = c.copy()
        # 根据官方指定的行列号范围做修正
        l_temp[c > self.c_max[res]] = -1
        l_temp[l > self.l_max[res]] = -1
        c_temp[c > self.c_max[res]] = -1
        c_temp[l > self.l_max[res]] = -1
        l_temp[c < 0] = -1
        l_temp[l < 0] = -1
        c_temp[c < 0] = -1
        c_temp[l < 0] = -1
        return l_temp, c_temp

    def get_flag(self, data: ndarray, ori_lat: ndarray, ori_lon: ndarray, l_bias: int = 0, c_bias: int = 0)\
            -> Tuple[ndarray, ndarray, ndarray]:
        # 根据经纬度矩阵，去除无效范围，设置flag标记有效值的区域，无效为0，有效为1
        ori_lat, ori_lon = self.fix_lat_lon(ori_lat, ori_lon)
        hh, ww = data.shape
        hh += l_bias
        ww += c_bias
        lat = ori_lat[l_bias:hh, c_bias:ww]
        lon = ori_lon[l_bias:hh, c_bias:ww]
        flag = np.ones(data.shape)
        flag[np.isnan(lon)] = 0
        flag[np.isnan(lat)] = 0
        return flag, lat, lon

    def gen_new_data(self, data: ndarray, ori_lat: ndarray, ori_lon: ndarray, new_lat: ndarray, new_lon: ndarray,
                     flag: ndarray, method: str, sv_path: str = '') -> ndarray:
        """
        插值得到所需的数据

        Args:
            data:       原始数据
            ori_lat:    原始的纬度矩阵，维度为(m, n)
            ori_lon:    原始的经度矩阵，维度为(m, n)
            new_lat:    需要插值到的纬度数列，维度为(m1, )
            new_lon:    需要插值到的经度数列，维度为(n1, )
            flag:       标记值，0为不需要参与插值的点，1为有效的需要参与插值的点
            method:     插值方法
            sv_path:    存储路径
        Returns:
            维度为(m1, n1)的数据块
        """
        new_lat_2d, new_lon_2d = np.meshgrid(new_lat, new_lon)
        new_lon_2d = new_lon_2d.T
        new_lat_2d = new_lat_2d.T
        values = data[flag > 0]
        points = np.c_[ori_lat[flag > 0], ori_lon[flag > 0]]
        new_data = ip.griddata(points, values, (new_lat_2d, new_lon_2d), method=method)
        if sv_path != '':
            utils.gen_nc_simple(new_data, new_lat, new_lon, sv_path)
        return new_data

    def raw_to_lat_lon(self, raw_path: str) -> Tuple[ndarray, ndarray]:
        """
        根据raw文件返回经纬度矩阵

        Args:
            raw_path: raw文件路径

        Returns:
            和raw文件一样大小的经纬度矩阵

        Notes:
            解码表：
                                   二进制文件解码（python）
            ———————————————————————————————————————————————————————————————-————
            |    FORMAT     |            TYPE           |     STANDARD SIZE    |
            |——————————————————————————————————————————————————————————————-———|
            |       c       |     string of length 1    |           1          |
            |       b       |          integer	        |           1          |
            |       B       |          integer	        |           1          |
            |       ?       |           bool 	        |           1          |
            |       h       |          integer	        |           2          |
            |       H       |          integer	        |           2          |
            |       i       |          integer	        |           4          |
            |       I       |          integer	        |           4          |
            |       l       |          integer	        |           4          |
            |       L       |          integer	        |           4          |
            |       q       |          integer	        |           8          |
            |       Q       |          integer	        |           8          |
            |       f       |           float	        |           4          |
            |       d       |           float	        |           8          |
            ————————————————————————————————————————————-———————————————————————
        """
        logging.debug("读取raw文件：{}".format(raw_path))
        res = raw_path.split(os.sep)[-1].split('_')[-1].split('.')[0] + 'm'
        logging.debug("文件对应的分辨率为：{}".format(res))
        with open(raw_path, 'rb') as f:
            f.seek(0)
            num = (self.l_max[res] + 1) ** 2                                # 总共的数据点数
            data = struct.unpack(str(2 * num) + 'd', f.read(2 * 8 * num))   # 2 * num个经纬度数据，每个8位
        data = np.array(list(data)).reshape(num, 2)
        # raw二进制文件中，每个网格对应16字节，前8字节为经度值，后8字节为纬度值，16km的raw是反过来的
        ori_lat = data[:, 0].reshape((self.l_max[res] + 1), (self.l_max[res] + 1))
        ori_lon = data[:, 1].reshape((self.l_max[res] + 1), (self.l_max[res] + 1))
        return ori_lat, ori_lon

    def lat_lon_to_l_c(self, new_lat: ndarray, new_lon: ndarray, l_bias: int, c_bias: int, res: str) \
            -> Tuple[ndarray, ndarray]:
        """
        根据指定的经纬度网格，获取数据

        Args:
            data:       原始数据块
            new_lat:    要插值到的纬度数列，m维向量
            new_lon:    要插值到的经度数列，n维向量
            l_bias:     行偏移，即：(标称投影中的第l行，相当于data中的 (l - l_bias) 行)
            c_bias:     列偏移，即：(标称投影中的第c列，相当于data中的 (c - c_bias) 列)
            res:        原始数据的分辨率
        Returns:
            指定经纬度网格下的行列矩阵
        """
        # 转为经纬度矩阵
        lat, lon = np.meshgrid(new_lat, new_lon)
        lon = lon.T
        lat = lat.T

        # 根据文档，从经纬度矩阵转成行列号矩阵
        lon = lon * self.pi / 180
        lat = lat * self.pi / 180
        lambda_d = self.lambda_d * self.pi / 180
        lambda_e = lon
        phi_e = np.arctan(np.tan(lat) * (self.eb ** 2) / (self.ea ** 2))
        r_e = self.eb / np.sqrt(1 - ((((self.ea ** 2) - (self.eb ** 2)) / (self.ea ** 2)) * (np.cos(phi_e) ** 2)))
        r1 = self.h - r_e * np.cos(phi_e) * np.cos(lambda_e - lambda_d)
        r2 = -1 * r_e * np.cos(phi_e) * np.sin(lambda_e - lambda_d)
        r3 = r_e * np.sin(phi_e)
        r_n = np.sqrt((r1 ** 2) + (r2 ** 2) + (r3 ** 2))
        x = np.arctan(-1 * r2 / r1) * (180 / self.pi)
        y = np.arctan(-1 * r3 / r_n) * (180 / self.pi)
        c = self.COFF[res] + x * (2 ** (-1 * 16)) * self.CFAC[res]
        l = self.LOFF[res] + y * (2 ** (-1 * 16)) * self.LFAC[res]
        l, c = (np.round(l).astype(int), np.round(c).astype(int))

        # 从标称投影的行列号还原到数据的行列号，并根据有效范围进行筛选
        l -= l_bias
        c -= c_bias
        l, c = self.fix_l_c(l, c, res)

        return l, c

    def l_c_to_Lat_lon(self, data: ndarray, l_bias: int, c_bias: int, **kwargs) -> Tuple[ndarray, ndarray]:
        """
        根据data的行列，计算得到data每个数据点的经纬度，并插值到指定经纬度网格，存储为nc数据

        Args:
            data:       读取到的行列数据
            l_bias:     行偏移，即：(data的第l行，相当于标称投影中的 (l + l_bias) 行)
            c_bias:     列偏移，即：(data的第c列，相当于标称投影中的 (c + c_bias) 列)
        Returns:
            原始数据块对应的经纬度矩阵
        """

        # 根据原始数据块得到行列矩阵
        hh, ww = data.shape
        if kwargs.get('l') is not None and kwargs.get('c') is not None and kwargs['l'].ndim == 2 and \
                kwargs['l'].ndim == 2:
            l = kwargs['l']
            c = kwargs['c']
        else:
            l = np.repeat(np.array([np.arange(hh)]).T, ww, axis=1) + l_bias
            c = np.repeat(np.array([np.arange(ww)]), hh, axis=0) + c_bias
        if kwargs.get('res') is None:
            kwargs['res'] = '4000m'
        # Step1.求 x,y
        x = (self.pi * (c - self.COFF[kwargs['res']])) / (180 * (2 ** (-1 * 16)) * self.CFAC[kwargs['res']])
        y = (self.pi * (l - self.LOFF[kwargs['res']])) / (180 * (2 ** (-1 * 16)) * self.LFAC[kwargs['res']])

        # Step2.求 sd,sn,s1,s2,s3,sxy
        s_d = np.sqrt(((self.h * np.cos(x) * np.cos(y)) ** 2) -
                      ((((np.cos(y)) ** 2) + ((self.ea ** 2) * (np.sin(y) ** 2) / (self.eb ** 2))) * ((self.h ** 2) -
                                                                                           (self.ea ** 2))))
        s_n = (self.h * np.cos(x) * np.cos(y) - s_d) / (((np.cos(y)) ** 2) +
                                                        ((self.ea ** 2) * (np.sin(y) ** 2) / (self.eb ** 2)))
        s1 = self.h - s_n * np.cos(x) * np.cos(y)
        s2 = s_n * np.sin(x) * np.cos(y)
        s3 = -1 * s_n * np.sin(y)
        s_xy = np.sqrt((s1 ** 2) + (s2 ** 2))

        # Step3 求原始的lon,lat
        ori_lon = (180 / self.pi) * np.arctan(s2 / s1) + self.lambda_d
        ori_lat = (180 / self.pi) * np.arctan(((self.ea ** 2) / (self.eb ** 2)) * (s3 / s_xy))
        return ori_lat, ori_lon

    def read_FY4A(self, file_path: str):
        """
        读取FY4A数据。添加经纬度信息（包括中国区和全圆盘），再转存为nc文件
        Args:
            file_path:  FY4A文件路径
            **kwargs:   读取数据的方法、插值方法
        """
        info = self.get_info_from_file_path(file_path)  # 获取基本信息
        if (info['Sensor_Name'] == 'AGRI') and (info['Format'] == 'NC'):
            # AGRI的L2产品
            return self.read_AGRI_L2(file_path)
        elif (info['Sensor_Name'] == 'AGRI') and (info['Format'] == 'HDF') and (info['Product_Name'] != 'GEO'):
            # AGRI的L1反射率产品
            return self.read_AGRI_L1_REF(file_path)
        elif (info['Sensor_Name'] == 'AGRI') and (info['Product_Name'] == 'GEO'):
            # AGRI的L1GEO产品
            return self.read_AGRI_L1_GEO(file_path)
        elif info['Sensor_Name'] == 'GIIRS':
            return self.read_GIIRS(file_path)
        elif info['Sensor_Name'] == 'LMI':
            return self.read_LMI(file_path)
        else:
            self.raiseE('输入的文件错误！')

    def read_AGRI_L1_REF(self, file_path: str) -> xr.Dataset:
        # FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_0500M_V0001.HDF
        # FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_0500M_V0001.HDF
        # FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_1000M_V0001.HDF
        # FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_1000M_V0001.HDF
        # FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_2000M_V0001.HDF
        # FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_2000M_V0001.HDF
        # FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.HDF
        # FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.HDF
        ds = xr.open_dataset(file_path, engine='netcdf4')
        data = ds['NOMChannel02'].values
        l_bias = ds.attrs['Begin Line Number']
        c_bias = ds.attrs['Begin Pixel Number']
        # 根据行列获得原始数据的经纬点
        ori_lat, ori_lon = self.l_c_to_Lat_lon(data, l_bias, c_bias)
        # 去除无效范围，设置flag标记有效值的区域，无效为0，有效为1
        flag, ori_lat, ori_lon = self.get_flag(data, ori_lat, ori_lon)
        ds['flag'] = (ds['NOMChannel02'].dims, flag)
        ds['lat'] = (ds['NOMChannel02'].dims, ori_lat)
        ds['lon'] = (ds['NOMChannel02'].dims, ori_lon)
        return ds

    def read_AGRI_L1_GEO(self, file_path: str) -> xr.Dataset:
        # FY4A-_AGRI--_N_DISK_1047E_L1-_GEO-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.HDF
        # FY4A-_AGRI--_N_REGC_1047E_L1-_GEO-_MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.HDF
        ds = xr.open_dataset(file_path, engine='netcdf4')
        l = ds['ColumnNumber'].values
        c = ds['LineNumber'].values
        data = ds['NOMSatelliteZenith'].values
        # 根据行列获得原始数据的经纬点
        ori_lat, ori_lon = self.l_c_to_Lat_lon(data, 0, 0, l=l, c=c)
        # 去除无效范围，设置flag标记有效值的区域，无效为0，有效为1
        flag, ori_lat, ori_lon = self.get_flag(data, ori_lat, ori_lon)
        ds['flag'] = (ds['ColumnNumber'].dims, flag)
        ds['lat'] = (ds['ColumnNumber'].dims, ori_lat)
        ds['lon'] = (ds['ColumnNumber'].dims, ori_lon)
        return ds

    def read_AGRI_L2(self, file_path: str) -> xr.Dataset:
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _ACI - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_1000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _AMV - _C009_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_064KM_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _AMV - _C010_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_064KM_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _AMV - _C012_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_064KM_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CFR - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CIX - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CLM - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CLP - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CLT - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CTH - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CTP - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _CTT - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _DLR - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _DSD - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _FHS - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_2000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _FOG - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _LPW - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _LSE - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_012KM_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _LST - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _OLR - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _QPE - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _RSR - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _SSI - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _SST - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _TBB - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _TFP - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_DISK_1047E_L2 - _ULR - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_NHEM_1047E_L2 - _CFR - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _ACI - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_1000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _CIX - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _CLM - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _CLP - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _CLT - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _CTH - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _CTP - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _CTT - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _DSD - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _FHS - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_2000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _LPW - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _QPE - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _TBB - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        # FY4A - _AGRI - -_N_REGC_1047E_L2 - _TFP - _MULT_NOM_yyyymmddHHMMSS_yyyymmddHHMMSS_4000M_V0001.NC
        Product_Name = self.get_info_from_file_path(file_path)['Product_Name']
        ds = xr.open_dataset(file_path)
        l_bias = ds['geospatial_lat_lon_extent'].attrs['begin_line_number']
        c_bias = ds['geospatial_lat_lon_extent'].attrs['begin_pixel_number']
        if self.product_name.get(Product_Name) is not None:
            Product_Name = self.product_name[Product_Name]
        data = ds[Product_Name].values[0] if 'LSE' in file_path else ds[Product_Name].values
        if 'AMV' in file_path:
            ds = xr.open_dataset(file_path)
            lat = ds['lat'].values
            lon = ds['lon'].values
            flag, ori_lat, ori_lon = self.get_flag(data, lat, lon)
        else:
            # 根据行列获得原始数据的经纬点
            ori_lat, ori_lon = self.l_c_to_Lat_lon(data, l_bias, c_bias)
            # 去除无效范围，设置flag标记有效值的区域，无效为0，有效为1
            flag, ori_lat, ori_lon = self.get_flag(data, ori_lat, ori_lon)
        dims = ds[Product_Name].dims[1:] if 'LSE' in file_path else ds[Product_Name].dims
        ds['flag'] = (dims, flag)
        ds['lat'] = (dims, ori_lat)
        ds['lon'] = (dims, ori_lon)
        return ds

    def read_LMI(self, file_path: str) -> xr.Dataset:
        # FY4A-_LMI---_N_REGX_1047E_L2-_LMIE_SING_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_7800M_NnnV1.NC
        # FY4A-_LMI---_N_REGX_1047E_L2-_LMIG_SING_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_7800M_NnnV1.NC
        ds = xr.open_dataset(file_path)
        ds = utils.rename_ds(ds, 'LAT', 'lat')
        ds = utils.rename_ds(ds, 'LON', 'lon')
        return ds

    def read_GIIRS(self, file_path: str) -> xr.Dataset:
        # FY4A-_GIIRS-_N_REGX_1047E_L1-_IRD-_MULT_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_016KM_nnnVm.HDF
        # FY4A-_GIIRS-_N_REGX_1047E_L2-_AVP-_MULT_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_016KM_nnnVm.NC
        # FY4A - _GIIRS - _N_REGX_1047E_L2 - _AVP - _MULT_NUL_yyyymmddHHMMSS_yyyymmddHHMMSS_016KM_V0002.NC
        if 'HDF' in file_path:
            return xr.open_dataset(file_path, engine='netcdf4')
        if 'NC' in file_path:
            return xr.open_dataset(file_path)

    def get_data_from_data_name(self, file_path: str, data_name: str, new_lon: ndarray, new_lat:ndarray,
                                method: str = 'nearest', sv_path: str = '') -> ndarray:
        ds = self.read_FY4A(file_path)
        ori_lat, ori_lon, flag, data = ds['lat'].values, ds['lon'].values, ds['flag'].values, ds[data_name].values
        return self.gen_new_data(data, ori_lat, ori_lon, new_lat, new_lon, flag, method, sv_path)


arg = components.Arg()
arg_parsed = arg.arg_parse('-d'.split())  # 代码输入参数
# arg_parsed = arg.arg_parse()  # 命令行输入参数

fy4a = FY4A(arg_parsed)


# FY4A文件路径(任写一种产品即可)
file_path = r'/home/developer_13/FY4A/codes/test_FY4A_data/' \
            r'FY4A-_AGRI--_N_DISK_1047E_L2-_CLT-_MULT_NOM_20210923010000_20210923011459_4000M_V0001.NC'
# 存储nc的路径
sv_path = r'/home/developer_13/FY4A/codes/CLT.nc'
# 需要插值的经纬度
new_lon, new_lat = utils.gen_lat_lon(70, 140, 60, 0, 0.04)
# 得到数据块，并存储为nc
fy4a.get_data_from_data_name(file_path, 'CLT', new_lon, new_lat, sv_path=sv_path)
# 数据对象
logging.debug(fy4a.read_FY4A(file_path))