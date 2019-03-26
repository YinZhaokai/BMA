# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

from datafomate import ECEns, ECReanalysis, GFSFcst, GEFSFcst, BMA, NCData
import numpy as np
import time
import warnings
import arrow
import os


def train_prepare(data_path, time_list, region):
    products = ['ec_fcst', 'gfs_fcst', 'ec_reanlys', 'gefs_fcst']
    for product in products[:2]:
        os.system('rm {}*'.format(data_path[product] + 'extract/'))
        for time in time_list[:]:
            # print(time)
            base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
            # --训练数据处理
            if product == 'ec_fcst':
                train = ECEns(data_path[product], time, base_path)
            elif product == 'gfs_fcst':
                train = GFSFcst(data_path[product], time, base_path)
            elif product == 'gefs_fcst':
                train = GEFSFcst(data_path[product], time, base_path)
            else:
                train = ECReanalysis(data_path[product], time, base_path)
            uv_files = train.download()
            train.wind_composite(uv_files)
        train_all = NCData(data_path[product], 'enforced')
        merge_file, ens_num = train_all.merge_time(time_list)
        train_all.extract_all(merge_file, region)


def fcst_prepare(data_path, time_list, region):
    products = ['ec_fcst', 'gfs_fcst', 'gefs_fcst']
    model_info = {}
    for product in products[:2]:
        for time in time_list[:]:
            # print(time)
            base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
            # --预报数据处理
            if product == 'ec_fcst':
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
            model_info[product] = ens_num
            fcst_all.extract_all(merge_file, region)
    print(model_info)
    return model_info


def bma_method(origin_path, time_list, region, train_num, all_num, model_info):
    single_bma = BMA(origin_path, region)
    single_bma.process_all(train_num, all_num, model_info)
    single_bma.point2nc(time_list)


def main():
    warnings.filterwarnings('ignore')
    # --路径参数
    data_path = '/home/qxs/bma/data/'
    model_path = {'ec_fcst': data_path + 'ecmwf_fcst/',
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
    train_num = 120   # 训练数据长度
    shift_hours = range(0, 96 + 24, 24)  # 预报时间间隔
    start = arrow.get('2018081412', 'YYYYMMDDHH')   # 预报起始时间
    for num in range(100):
        ini_time = start.shift(hours=num*12)
        # --生成训练时间列表、预报时间列表
        train_list = []
        fcst_list = []
        for num in range(1, train_num + 1)[::-1]:
            train_list.append({'ini': arrow.get(ini_time).shift(hours=-num*12), 'shift': 0})
        for shift_hour in shift_hours[:]:
            fcst_list.append({'ini': arrow.get(ini_time), 'shift': shift_hour})
        # -- 数据处理
        file_time = fcst_list[0]['ini'].format('YYYYMMDDHH')
        bma_file = model_path['bma_fcst'] + file_time + '/ws_{}.nc'.format(file_time)
        if not os.path.exists(bma_file):
            train_prepare(model_path, train_list, region)
            model_info = fcst_prepare(model_path, fcst_list, region)
            # model_info = {'ec_fcst': 51, 'gfs_fcst':1}
            bma_method(model_path, fcst_list, region, train_num, train_num + len(shift_hours), model_info)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    delta_t = end - start
    print(delta_t)
