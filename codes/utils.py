# coding: UTF-8
"""
@FileName: utils.py
@author：shi yao  <aiwei169@sina.com>
@date: 2021/4/28 13:49
@Software: PyCharm
python version: python 3.6.13
"""

"""
通用函数组件
"""

import numpy as np
from numpy import ndarray
from typing import Tuple
import matplotlib.pyplot as plt
import scipy.interpolate as ip
import datetime as dt
import pickle
import xarray as xr
import pandas as pd
import os
import logging
import warnings
warnings.filterwarnings('ignore')


def plot(matrix: ndarray, save_path: str = ''):
    plt.imshow(matrix)
    plt.axis('off')
    if save_path != '':  # 如果保存图片，就不在控制台展示了
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def calc_scale_and_offset(min_v, max_v, n) -> Tuple[ndarray, ndarray]:
    """生成压缩netcdf数据需要用的参数

    Args:
        min_v: 数据最小值
        max_v: 数据最大值
        n: 指定位长，比如16位、8位

    Returns: 指定参数
    """
    if max_v - min_v == 0:
        scale_factor = 1.0
        add_offset = min_v
    else:
        scale_factor = (max_v - min_v) / (2 ** n - 1)
        add_offset = min_v + 2 ** (n - 1) * scale_factor
    return scale_factor, add_offset

def interpolation(data: ndarray, lat: ndarray, lon: ndarray, new_lat: ndarray, new_lon: ndarray,
                  method: str = 'linear') -> ndarray:
    # 插值
    if lat[1] < lat[0]:
        lat = lat[::-1]
        data = np.flip(data, 0)
    if lon[1] < lon[0]:
        lon = lon[::-1]
        data = np.flip(data, 1)
    XX, YY = np.meshgrid(new_lat, new_lon)
    XX = XX.T
    YY = YY.T
    x_, y_ = XX.shape
    new_data = ip.interpn((lat, lon), data, np.c_[XX.reshape(x_ * y_), YY.reshape(x_ * y_)],
                          method=method, bounds_error=False, fill_value=None)  # 允许外推，允许输入有nan
    new_data = new_data.reshape(x_, y_)
    return new_data

def interpolation_3D(data: ndarray, lat: ndarray, lon: ndarray, new_lat: ndarray, new_lon: ndarray,
                  method: str = 'linear') -> ndarray:
    # 3D插值，第一个维度是时间
    new_data = np.zeros((len(data), len(new_lat), len(new_lon)))
    for i in range(len(data)):
        data_temp = data.copy()[i, :, :]
        new_data[i, :, :] = interpolation(data_temp, lat.copy(), lon.copy(), new_lat.copy(), new_lon.copy(), method)
    return new_data


def interp(data: ndarray, lon: ndarray, lat: ndarray, new_lon: ndarray, new_lat: ndarray, method='linear') -> ndarray:
    lon_ = lon.flatten()
    lat_ = lat.flatten()
    lon_lat = np.array([lon_, lat_]).T
    dat = np.array(data).flatten()

    n_lon, n_lat = np.meshgrid(new_lon, new_lat)
    n_lon_ = n_lon.flatten()
    n_lat_ = n_lat.flatten()
    n_lon_lat = np.array([n_lon_, n_lat_]).T

    new_data = ip.griddata(lon_lat, dat, n_lon_lat, method=method).reshape(len(new_lat), len(new_lon))
    return new_data

def adj_m(m: ndarray, max: float, min: float) -> ndarray:
    # 把矩阵m放缩到[min, max]之间
    if m[np.isnan(m)==False].size == 0:
        return m
    return (m - np.nanmin(m)) / (np.nanmax(m) - np.nanmin(m)) * (max - min) + min

def wall_side(m: ndarray, big: bool = Tuple) -> ndarray:
    # 把矩阵最外面的一圈转为最大或者最小的值
    if m[np.isnan(m)==False].size == 0:
        return m
    fix = np.nanmax(m) + 1 if big else np.nanmin(m) - 1
    m[0, :] = fix
    m[-1:, :] = fix
    m[:, 0] = fix
    m[:, -1:] = fix
    return m

def add_to_dict_3d(update_dict, key_a, key_b, key_c, val) -> None:
    # 给字典增加三维元素
    if key_a in update_dict:
        if key_b in update_dict[key_a]:
            update_dict[key_a][key_b].update({key_c: val})
        else:
            update_dict[key_a].update({key_b: {key_c: val}})
    else:
        update_dict.update({key_a: {key_b: {key_c: val}}})

def add_to_dict_2d(update_dict, key_a, key_b, val) -> None:
    # 给字典增加二维元素
    if key_a in update_dict:
        update_dict[key_a].update({key_b: val})
    else:
        update_dict.update({key_a: {key_b: val}})

def add_to_dict_1d(update_dict, key_a, val) -> None:
    # 给字典增加一维元素
    update_dict[key_a] = val

def update_time_integral_1hour(t: dt.datetime) -> dt.datetime:
    # 把时刻修正到整点
    return dt.datetime(t.year, t.month, t.day, t.hour, 0, 0)

def update_time_integral_3hour(t: dt.datetime) -> dt.datetime:
    # 把时刻修正到整3小时的点
    return dt.datetime(t.year, t.month, t.day, t.hour - t.hour % 3, 0)

def update_time_08_20(t: dt.datetime) -> dt.datetime:
    # 把时刻修正到08、20
    if t.hour < 8:
        fixed_t =  dt.datetime(t.year, t.month, t.day, 8, 0)
    elif (t.hour >= 8) & (t.hour < 20):
        fixed_t = dt.datetime(t.year, t.month, t.day, 20, 0)
    else:
        t += dt.timedelta(days=1)
        fixed_t = dt.datetime(t.year, t.month, t.day, 8, 0)
    return fixed_t - dt.timedelta(hours=12)

def update_time_02_08_14_20(t: dt.datetime) -> dt.datetime:
    # 把时刻修正到02、08、14、20
    if t.hour < 2:
        fixed_t =  dt.datetime(t.year, t.month, t.day, 2, 0)
    elif (t.hour >= 2) & (t.hour < 8):
        fixed_t = dt.datetime(t.year, t.month, t.day, 8, 0)
    elif (t.hour >= 8) & (t.hour < 14):
        fixed_t = dt.datetime(t.year, t.month, t.day, 14, 0)
    elif (t.hour >= 14) & (t.hour < 20):
        fixed_t = dt.datetime(t.year, t.month, t.day, 20, 0)
    else:
        t += dt.timedelta(days=1)
        fixed_t =   dt.datetime(t.year, t.month, t.day, 2, 0)
    return fixed_t - dt.timedelta(hours=6)

def update_time_integral_10min(t: dt.datetime) -> dt.datetime:
    # 把时刻修正到整10分钟
    return dt.datetime(t.year, t.month, t.day, t.hour, t.minute - t.minute % 10, 0)  # 定位到所在的整10分钟

def update_time_integral_6min(t: dt.datetime) -> dt.datetime:
    # 把时刻修正到整6分钟
    return dt.datetime(t.year, t.month, t.day, t.hour, t.minute - t.minute % 6, 0)  # 定位到所在的整6分钟

def time2str(dt_time: dt, time_format: str) -> str:
    # datetime转str
    return dt.datetime.strftime(dt_time, time_format)

def str2time(time_str: str, time_format: str) -> dt:
    # str转datetime
    return dt.datetime.strptime(time_str, time_format)

def bjt2utc(dt_time: dt) -> dt:
    # 北京时间转UTC时间
    return dt_time - dt.timedelta(hours=8)

def utc2bjt(dt_time: dt) -> dt:
    # UTC时间转北京时间
    return dt_time + dt.timedelta(hours=8)

def gen_72_time_list_r(t: dt) -> list:
    # 生成从t时刻往前72小时逐小时的时刻列表
    start = t - dt.timedelta(hours=71)
    time_list = []
    while start <= t:
        time_list.append(start)
        start = start + dt.timedelta(hours=1)
    return time_list

def gen_240_time_list_f(t: dt) -> list:
    # 生成从t时刻往后240小时逐3小时的时刻列表
    start = t
    end = t + dt.timedelta(hours=239)
    time_list = []
    while start <= end:
        time_list.append(start)
        start = start + dt.timedelta(hours=3)
    return time_list

def sv(data, name):
    # 单个变量保存为name.pckl
    f = open(name + '.pckl', 'wb')
    pickle.dump(data, f)
    f.close()

def read(pckl_path):
    # 读取pckl
    f = open(pckl_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def is_nc_exist(save_path: str) -> bool:
    if os.path.exists(save_path):
        try:
            xr.open_dataset(save_path)
            logging.debug("{}已存在，不重复生成".format(save_path))
            return False
        except:
            logging.debug("{}生成错误，重新生成".format(save_path))
            os.remove(save_path)
            return True
    else:
        logging.debug("数据不存在需要生成".format(save_path))
        return True

def gen_nc(data: ndarray, lat: ndarray, lon: ndarray, product_time: dt, gap: Tuple[int, int, int, int, int, int, int],
           save_path: str, freq:str, data_name: str = 'tp', data_long_name: str = 'Total precipitation',
           units: str = 'mm', dtype: str = 'int16', _FillValue: int = 0):
    gap0_1, gap0_2, gap1_1, gap1_2, gap2_1, gap2_2, period = gap
    ds = xr.Dataset()
    pd_time_list = pd.date_range(product_time + dt.timedelta(hours=gap0_1, minutes=gap0_2), freq=freq, periods=period)
    if gap0_2 == 0:
        time_list = np.array((pd_time_list - product_time) / 3600e9).astype(int)
    else:
        time_list = np.array((pd_time_list - product_time) / 60e9).astype(int)
    ds['time'] = ('time', time_list)
    ds['time'].attrs['tips'] = data_long_name + " between {} and {}".\
        format(dt.datetime.strftime(product_time + dt.timedelta(hours=gap1_1, minutes=gap1_2), '%Y-%m-%d %H:%M:%S'),
               dt.datetime.strftime(product_time + dt.timedelta(hours=gap2_1, minutes=gap2_2), '%Y-%m-%d %H:%M:%S'))
    ds['time'].attrs['long_name'] = "Time(CST)"
    if gap0_2 == 0:
        ds['time'].attrs['units'] = 'hours since ' + product_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        ds['time'].attrs['units'] = 'minutes since ' + product_time.strftime('%Y-%m-%d %H:%M:%S')
    ds.coords['lat'] = ('lat', lat)
    ds['lat'].attrs['units'] = "degrees_north"
    ds['lat'].attrs['long_name'] = "Latitude"
    ds.coords['lon'] = ('lon', lon)
    ds['lon'].attrs['units'] = "degrees_east"
    ds['lon'].attrs['long_name'] = "Longitude"
    ds[data_name] = (('time', 'lat', 'lon'), data.reshape((period, len(lat), len(lon))))
    ds.attrs['CreateTime'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds[data_name].attrs['units'] = units

    # 存储
    scale_factor, add_offset = calc_scale_and_offset(np.nanmin(data), np.nanmax(data), 16)
    ds.to_netcdf(
        save_path,
        engine='netcdf4',
        encoding={
            data_name: {
                'dtype': dtype,
                'scale_factor': scale_factor,
                'complevel': 9,
                'zlib': True,
                '_FillValue': _FillValue,
                'add_offset': add_offset
            }
        }
    )

def gen_nc_simple(data: ndarray, lat: ndarray, lon: ndarray, save_path: str):
    ds = xr.Dataset()
    ds.coords['lat'] = ('lat', lat)
    ds.coords['lon'] = ('lon', lon)
    ds['var'] = (('lat', 'lon'), data.reshape((len(lat), len(lon))))
    ds.to_netcdf(save_path)

def ispath(path: str):
    # 判断路径是否存在，不存在就生成这个路径
    if not os.path.exists(path):
        os.makedirs(path)

def rename_ds(ds: xr.Dataset, name: str, new_name: str) -> xr.Dataset:
    # 修改dataset的数据名字
    return ds.rename({name: new_name})

def del_ds_attrs(ds: xr.Dataset, attr_name: str) -> xr.Dataset:
    # 删除dataset的某个属性
    del ds.attrs[attr_name]
    return ds

def gen_lat_lon(l_lon: float, r_lon: float, t_lat: float, b_lat: float, res: float) -> Tuple[ndarray, ndarray]:
    return np.arange(l_lon, r_lon + res / 2, res), np.arange(t_lat, b_lat - res / 2, -1 * res)

def gen_new_lat_lon(lat: ndarray, lon: ndarray, res: float) -> Tuple[ndarray, ndarray]:
    l_lon, r_lon, t_lat, b_lat = float(np.nanmin(lon)), float(np.nanmax(lon)), \
                                 float(np.nanmax(lat)), float(np.nanmin(lat))
    return gen_lat_lon(l_lon, r_lon, t_lat, b_lat, res)