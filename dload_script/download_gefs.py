# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import sys
sys.path.append("/home/qxs/bma/dload_script/")
from datadl_pack import GEFSFcst
import arrow
from pathos.pools import ProcessPool


def multi_process(data_path, time_list):
    for time in time_list[:]:
        # print(time)
        base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
        # --预报数据处理
        gefs_fcst = GEFSFcst(data_path['gefs_fcst'], time, base_path)
        p = ProcessPool(7)
        for n in range(21):
            # gefs_fcst.download(n)
            p.apipe(download, gefs_fcst, n)
        p.close()
        p.join()
        p.clear()


def download(instance, n):
    instance.download(n)


def main():
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
    # ini_time = arrow.get('2019040900', 'YYYYMMDDHH')
    shift_hours = range(0, 96 + 6, 6)  # 预报时间间隔
    # --生成训练时间列表、预报时间列表
    fcst_list = []
    for shift_hour in shift_hours[:]:
        fcst_list.append({'ini': ini_time, 'shift': shift_hour})
    # -- 数据处理
    multi_process(model_path, fcst_list)

if __name__ == '__main__':
    main()
