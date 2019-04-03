# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import sys
sys.path.append("/home/qxs/bma")
from ncprocess import GEFSFcst
import arrow


def download(data_path, time_list):
    for time in time_list[:]:
        # print(time)
        base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
        # --预报数据处理
        gefs_fcst = GEFSFcst(data_path['gefs_fcst'], time, base_path)
        uv_files = []
        for n in range(21):
            uv_files.append(gefs_fcst.download(n))
        gefs_fcst.wind_composite(uv_files)


if __name__ == '__main__':
    # --路径参数
    data_path = '/home/qxs/bma/data/'
    model_path = {'ec_fcst': data_path + 'ecmwf_fcst/',
                       'ec_reanlys': data_path + 'ecmwf_reanlys/',
                       'gfs_fcst': data_path + 'gfs_fcst/',
                       'gefs_fcst': data_path + 'gefs_fcst/',
                       'bma_fcst': data_path + 'bma_result/'}
    # --时间参数
    time = arrow.get().now().date()   # 预报起始时间
    now = arrow.get().now().format('HH')
    if int(now) > 12:
        ini_time = arrow.get(time)
    else:
        ini_time = arrow.get(time).shift(hours=-12)
    # print(ini_time)
    # ini_time = arrow.get('2019040100', 'YYYYMMDDHH')
    shift_hours = range(0, 96 + 12, 12)  # 预报时间间隔
    # --生成训练时间列表、预报时间列表
    fcst_list = []
    for shift_hour in shift_hours[:]:
        fcst_list.append({'ini': ini_time, 'shift': shift_hour})
    # -- 数据处理
    download(model_path, fcst_list)
