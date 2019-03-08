# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

from datafomate import ECFcst, ECReanalysis, GFSFcst, GEFSFcst, BMA, NCData
import numpy as np
import time
import warnings
import arrow


def fcst_prepare(data_path, time_list, region):
    products = ['ec_fcst', 'gfs_fcst', 'gefs_fcst']
    for product in products[:2]:
        for time in time_list[:]:
            # print(time)
            base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
            # --预报数据处理
            if product == 'ec_fcst':
                fcst = ECFcst(data_path[product], time, base_path)
            else:
                fcst = GFSFcst(data_path[product], time, base_path)
            uv_files = fcst.download()
            fcst.wind_composite(uv_files)
        fcst_all = NCData(data_path[product])
        merge_file = fcst_all.merge_time(time_list)
        fcst_all.extract_all(merge_file, region)


def obs_paepare(data_path, time_list, region):
    for time in time_list[:]:
        # print(time)
        base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
        # --再分析数据处理
        ec_reanlys = ECReanalysis(data_path['ec_reanlys'], time, base_path)
        uv_file = ec_reanlys.download()
        ec_reanlys.wind_composite(uv_file)
    ec_reanlys_all = NCData(data_path['ec_reanlys'])
    merge_file = ec_reanlys_all.merge_time(time_list)
    ec_reanlys_all.extract_all(merge_file, region)


def bma_method(origin_path, time_list, region, train_num, all_num, ens_num1, ens_num2):
    single_bma = BMA(origin_path, region)
    single_bma.process_all(train_num, all_num, ens_num1, ens_num2)
    single_bma.point2grid(time_list)


def gefs(data_path, time_list, region):
    for time in time_list[:]:
        # print(time)
        base_path = arrow.get(time['ini']).format('YYYYMMDDHH')
        # --预报数据处理
        gefs_reanlys = GEFSFcst(data_path['gefs_fcst'], time, base_path)
        uv_files = []
        for n in range(21):
            uv_files.append(gefs_reanlys.download(n))
        gefs_reanlys.wind_composite(uv_files)

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
    region = zip(lat.flat[:], lon.flat[:])
    # --时间参数
    ini_time = '2019030100'   # 预报起始时间
    train_num = 120   # 训练数据数据长度
    shift_hours = range(0, 96 + 12, 12)  # 预报时间间隔
    # --生成训练时间列表、预报时间列表
    train_list = []
    fcst_list = []
    for num in range(1, train_num + 1)[::-1]:
        train_list.append({'ini': arrow.get(ini_time, 'YYYYMMDDHH').shift(hours=-num*12), 'shift': 0})
    for shift_hour in shift_hours[:]:
        fcst_list.append({'ini': arrow.get(ini_time, 'YYYYMMDDHH'), 'shift': shift_hour})
    # -- 数据处理
    file_time = fcst_list[0]['ini'].format('YYYYMMDDHH')
    bma_file = model_path['bma_fcst'] + file_time + '/ws_{}.nc'.format(file_time)
    gefs(model_path, fcst_list, region)
    # if not os.path.exists(bma_file):
    #     # fcst_prepare(model_path, train_list + fcst_list, region)
    #     # obs_paepare(model_path, train_list, region)
    #     bma_method(model_path, fcst_list, region, train_num, train_num + len(shift_hours), 50, 1)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    delta_t = end - start
    print(delta_t)
