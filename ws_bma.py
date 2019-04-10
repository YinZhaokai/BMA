# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import shutil
import numpy as np
import warnings
import arrow
import glob
import os
os.chdir('/home/qxs/bma/')
from ncprocess import ECEns, FNL, GFSFcst, GEFSFcst, BMA, NCData
import wind_map


def train_prepare(data_path, time_list, region, ini_time):
    products = ['fnl', 'gefs_fcst', 'ec_ens']
    group_info_train = {}
    for product in products[:3]:
        dirs = glob.glob(data_path[product] + 'extract_*')
        for dir in dirs:
            try:
                shutil.rmtree(dir)
            except OSError:
                pass
        for time in time_list[:]:
            # print(time)
            base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
            # --训练数据处理
            if product == 'ec_ens':
                train = ECEns(data_path[product], time, base_path)
            elif product == 'gefs_fcst':
                train = GEFSFcst(data_path[product], time, base_path)
            else:
                train = FNL(data_path[product], time, base_path)
            uv_files = train.download()
            train.wind_composite(uv_files)
        train_all = NCData(data_path[product], 'enforced')
        merge_file, ens_num = train_all.merge_time(time_list)
        if merge_file is not None:
            group_info_train[product] = ens_num
            train_all.extract_all(merge_file, region, ini_time)
    return group_info_train


def fcst_prepare(data_path, time_list, region, ini_time, group_info_train):
    products = ['gefs_fcst', 'ec_ens']
    products = list(set(group_info_train.keys()) & set(products))
    group_info_fcst = {}
    for product in products[:]:
        for time in time_list[:]:
            # print(time)
            base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
            # --预报数据处理
            if product == 'ec_ens':
                fcst = ECEns(data_path[product], time, base_path)
            elif product == 'gfs_fcst':
                fcst = GFSFcst(data_path[product], time, base_path)
            else:
                fcst = GEFSFcst(data_path[product], time, base_path)
            uv_files = fcst.download()
            fcst.wind_composite(uv_files)
        fcst_all = NCData(data_path[product], 'unenforced')
        merge_file, ens_num = fcst_all.merge_time(time_list)
        if merge_file is not None:
            group_info_fcst[product] = ens_num
            fcst_all.extract_all(merge_file, region, ini_time)
    return group_info_fcst


def bma_method(data_path, fcst_list, region, train_num, all_num, group_info_fcst):
    for fcst_time in fcst_list:
        dirs = glob.glob(data_path['bma_fcst'] + 'bma_point_*')
        for dir in dirs:
            try:
                shutil.rmtree(dir)
            except OSError:
                pass
        dirs = glob.glob(data_path['bma_fcst'] + 'extract*')
        for dir in dirs:
            try:
                shutil.rmtree(dir)
            except OSError:
                pass
        print('bma start... ')
        single_bma = BMA(data_path, fcst_time, region, group_info_fcst)
        single_bma.process_all(train_num, all_num)
        single_bma.point2nc()


def main():
    warnings.filterwarnings('ignore')
    # os.system('f2py -c module_bma.f90 main.f90 -m bma_method --fcompiler=intelem')
    # --路径参数
    data_path = '/home/qxs/bma/data/'
    model_path = {'fnl':  data_path + 'fnl/',
                        'ec_ens': data_path + 'ecmwf_ens/',
                        'ec_fine': data_path + 'ecmwf_fine/',
                        'ec_reanlys': data_path + 'ecmwf_reanlys/',
                        'gfs_fcst': data_path + 'gfs_fcst/',
                        'gefs_fcst': data_path + 'gefs_fcst/',
                        'bma_fcst': data_path + 'bma_result/'}
    # --空间参数
    region = {'lats': 0, 'latn': 42, 'lonw': 100, 'lone': 130}
    x = np.arange(region['lonw'], region['lone'] + 1, 0.5)   # 经度
    y = np.arange(region['lats'], region['latn'] + 1, 0.5)[::-1]   #维度
    lat, lon = np.meshgrid(y, x, indexing='ij')
    region = list(zip(lat.flat[:], lon.flat[:]))
    # --时间参数
    train_num = 60   # 训练数据长度
    interval = 6
    shift_hours = range(24, 96 + interval, interval)  # 预报时间间隔
    time = arrow.get().now().date()   # 预报起始时间
    now = arrow.get().now().format('HH')
    if int(now) > 12:
        ini_time = arrow.get(time).shift(days=-1, hours=12)
    else:
        ini_time = arrow.get(time).shift(days=-1)
    # ini_time = arrow.get('2019040800', 'YYYYMMDDHH')   # 预报起始时间
    for num in range(1):
        for shift_hour in shift_hours[:]:
            # --生成训练时间列表、预报时间列表
            train_list = []
            fcst_list = []
            for num in range(int(1 + (shift_hour - 24) / interval), int(train_num + 1 + (shift_hour - 24) / interval))[::-1]:
                train_list.append({'ini': ini_time.shift(hours=-num*12), 'shift': shift_hour})
            fcst_list.append({'ini': ini_time, 'shift': shift_hour})
            # -- 数据处理
            group_info_train = train_prepare(model_path, train_list, region, ini_time)
            group_info_fcst = fcst_prepare(model_path, fcst_list, region, ini_time, group_info_train)
            # # group_info = {'ec_ens': 51, 'gefs_fcst':21}
            print(group_info_train)
            print(group_info_fcst)
            bma_method(model_path, fcst_list, region, len(train_list), len(train_list) + len(fcst_list), group_info_fcst)
            wind_map.plot(model_path['bma_fcst'] + '{}/'.format(ini_time.format('YYYYMMDDHH')), ini_time, shift_hour)


if __name__ == '__main__':
    start = arrow.get().now()
    main()
    end = arrow.get().now()
    delta_t = end - start
    print(delta_t)
