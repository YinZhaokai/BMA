# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import pandas as pd
import xarray as xr
import glob
import os
import arrow
import warnings
import ftplib
import  sys
sys.path.append("/home/qxs/bma")
sys.path.append("/home/qxs/bma/dload_script")
import ncprocess
import datadl_pack


class FishCell(object):
    def __init__(self, ini_time, ec_ens_path, gefs_path, bma_ws, gfs_path):
        self.ini_time = ini_time
        self.ec_ens_path = ec_ens_path
        self.gefs_path = gefs_path
        self.bma_ws = bma_ws
        self.gfs_path = gfs_path

    def extract(self, shift_hours, info, index, code, lat, lon, out_file):
        for hour in shift_hours:
            # get wind speed
            if info['ws'] == 'bma':
                ws_name = 'ws_expect_{}_{}.nc'.format(self.ini_time.format('YYYYMMDDHH'), str(hour).zfill(2))
                ws = xr.open_dataset(self.bma_ws + ws_name)['ws'].sel(lat=lat, lon=lon, method='nearest').values[0]
            else:
                ws_name = 'ws_{}_{}.nc'.format(self.ini_time.format('YYYYMMDDHH'), str(hour).zfill(2))
                ws = xr.open_dataset(self.gfs_path + ws_name)['ws'].sel(latitude=lat, longitude=lon, method='nearest').values[0]
            # get wind direction
            ws_name = 'ws_{}_{}.nc'.format(self.ini_time.format('YYYYMMDDHH'), str(hour).zfill(2))
            if info['wd'] == 'gfs':
                wd = xr.open_dataset(self.gfs_path + ws_name)['wd'].sel(latitude=lat, longitude=lon, method='nearest').values[0]
            elif info['wd'] == 'gefs':
                wd = xr.open_dataset(self.gefs_path + ws_name)['wd'].sel(latitude=lat, longitude=lon, number=0, method='nearest').values[0]
            else:
                wd = xr.open_dataset(self.ec_ens_path + ws_name)['wd'].sel(latitude=lat, longitude=lon, number=0, method='nearest').values[0]
            # get time initial time format
            time = arrow.get(self.ini_time).to('Asia/Shanghai').shift(hours=24).format('YYYY-MM-DD HH:mm:ss')
            line = '{:4d}, {}, {}, {}, {:4.1f}, {:3.0f}'.format(index, code, time, str(hour - 24).zfill(2), ws, wd)
            with open(out_file, 'a') as f:
                f.write(line + '\n')


def celldata_produce(ini_time, shift_hours, wind_info, model_path, initime_path):
    # --路径信息
    fishzone_path = '/data2/fish_zone/'
    ec_ens_path = model_path['ec_ens'] + initime_path
    gefs_path = model_path['gefs_fcst'] + initime_path
    bma_path = model_path['bma_fcst'] + initime_path
    try:
        gfs_path = sorted(glob.glob(model_path['gfs_fcst'] + '*'))[-1] + '/'
    except IndexError:
        gfs_path = model_path['gfs_fcst'] + '*'
    for key in wind_info.keys():
        value = wind_info[key]
        if value == 'bma':
            print('{} use {}'.format(key, bma_path))
        elif value == 'gefs':
            print('{} use {}'.format(key, gefs_path))
        elif value == 'ec_ens':
            print('{} use {}'.format(key, ec_ens_path))
        else:
            print('{} use {}'.format(key, gfs_path))
    # --输出渔区信息
    if sorted(wind_info.keys()) == sorted(['ws', 'wd']):
        outpath =fishzone_path + '{}/'.format(ini_time.shift(days=1).format('YYYYMMDDHH'))
        try:
            os.makedirs(outpath)
        except OSError:
            pass
        out_file = outpath + 'FishCell_{}-{}-{}_SeaWind_{}.txt'.format(
            ini_time.format('YYYY'), ini_time.format('MM'), ini_time.shift(days=1).format('DD'), str(shift_hours[-1]-24))
        with open(out_file, 'a') as f:
            f.write('FISHERYCELL_SeaWind_V100' + '\n')
        info_file = fishzone_path + 'yuqu-new-20170503.txt'
        info = pd.read_csv(info_file, sep='\s+', header=None, index_col=0)
        info.columns = ['code', 'lat', 'lon']
        for index, row in list(info.iterrows())[:]:
            code = row['code']
            lat = row['lat']
            lon = row['lon']
            cell = FishCell(ini_time, ec_ens_path, gefs_path, bma_path, gfs_path)
            cell.extract(shift_hours, wind_info, index, code, lat, lon, out_file)
        with open(out_file, 'a') as f:
            f.write('END' + '\n')
        return out_file


def fishzone_product(model_path, ini_time, shift_hours):
    warnings.filterwarnings('ignore')
    # --下载备用gfs数据
    try:
        datadl_pack.GFSFcst().download(ini_time.shift(hours=12))
    except Exception as e:
        print('GFS can not get from remote server -> {}'.format(e))
    # --检查bma和gfs哪个数据源完整，优先使用bma算法结果
    initime_path = '{}/'.format(ini_time.format('YYYYMMDDHH'))
    ec_ens_path = model_path['ec_ens'] + initime_path
    gefs_path = model_path['gefs_fcst'] + initime_path
    bma_path = model_path['bma_fcst'] + initime_path
    wind_info = {}
    if set(glob.glob(bma_path + 'ws_expect*.nc')).issubset(set(glob.glob(bma_path + '*'))) and len(glob.glob(bma_path + 'ws_expect*.nc')) == len(shift_hours):
        print('use bma data to generate fish zone product')
        wind_info['ws'] = 'bma'
        if set(glob.glob(ec_ens_path + 'ws_*.nc')).issubset(set(glob.glob(ec_ens_path + '*'))) and len(glob.glob(ec_ens_path + 'ws_*.nc')) == len(shift_hours):
            wind_info['wd'] = 'ec_ens'
        elif set(glob.glob(gefs_path + 'ws_*.nc')).issubset(set(glob.glob(gefs_path + '*'))) and len(glob.glob(gefs_path + 'ws_*.nc')) == len(shift_hours):
            wind_info['wd'] = 'gefs'
        else:
            wind_info['wd'] = 'gfs'
    else:
        print('use gfs data to generate fish zone product')
        wind_info['ws'] = 'gfs'
        wind_info['wd'] = 'gfs'
        for shift_hour in shift_hours[:]:
            time = {'ini': ini_time, 'shift': shift_hour}
            gfs = ncprocess.GFSFcst(model_path['gfs_fcst'], time, initime_path)
            uv_files = gfs.download()
            gfs.wind_composite(uv_files)
    out_file = celldata_produce(ini_time, shift_hours, wind_info, model_path, initime_path)
    return out_file


def send_ftp(ip, username, password, local_file):
    port = 21
    ftp = ftplib.FTP()
    ftp.set_debuglevel(1)
    ftp.connect(ip, port)
    ftp.login(username, password)
    file = open(local_file, 'rb')
    ftp.storbinary('STOR ' + os.path.basename(local_file), file)
    file.close()
    ftp.quit()


if __name__ == '__main__':
    # --每天早上7:30执行该脚本
    # --路径参数
    data_path = '/home/qxs/bma/data/'
    model_path = {'fnl':  data_path + 'fnl/',
                        'ec_ens': data_path + 'ecmwf_ens/',
                        'ec_fine': data_path + 'ecmwf_fine/',
                        'ec_reanlys': data_path + 'ecmwf_reanlys/',
                        'gfs_fcst': data_path + 'gfs_fcst/',
                        'gefs_fcst': data_path + 'gefs_fcst/',
                        'bma_fcst': data_path + 'bma_result/'}
    # --时间参数
    date = arrow.get().now().date()   # 预报起始时间
    ini_time = arrow.get(date).shift(days=-1)
    shift_hours = range(24, 96 + 6, 6)
    # --生成渔区文件
    out_file = fishzone_product(model_path, ini_time, shift_hours)
    # out_file = '/data2/fish_zone/2019051700/FishCell_2019-05-17_SeaWind_72.txt'
    send_ftp('xxx.xxx.xxx.xx', 'xxxx', 'xxxxxxxx', out_file)
    send_ftp('xxx.x.xx.xx', 'xxxx', 'xxxxxxxx', out_file)
