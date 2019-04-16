# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import pandas as pd
import xarray as xr
import glob
import os
import arrow


class FishCell(object):
    def __init__(self, ini_time, ec_ens_path, gefs_path, bma_ws):
        self.ini_time = ini_time
        self.ec_ens_path = ec_ens_path
        self.gefs_path = gefs_path
        self.bma_ws = bma_ws

    def extract(self, shift_hours, info, index, code, lat, lon, out_file):
        for hour in shift_hours:
            # get wind speed
            ws_file = self.bma_ws + 'ws_expect*{}.nc'.format(str(hour).zfill(2))
            ws = xr.open_dataset(glob.glob(ws_file)[0])['ws'].sel(lat=lat, lon=lon, method='nearest').values[0]
            # get time initial time format
            time = arrow.get(self.ini_time).to('Asia/Shanghai').shift(hours=24).format('YYYY-MM-DD HH:mm:ss')
            # get wind direction
            uv_name = 'ws_{}_{}.nc'.format(self.ini_time.format('YYYYMMDDHH'), str(hour).zfill(2))
            if info['wd'] == 'ec_ens':
                wd = xr.open_dataset(self.ec_ens_path + uv_name)['wd'].sel(latitude=lat, longitude=lon, number=0, method='nearest').values[0]
            else:
                wd = xr.open_dataset(self.gefs_path + uv_name)['wd'].sel(latitude=lat, longitude=lon, number=0, method='nearest').values[0]
            line = '{:4d}, {}, {}, {}, {:4.1f}, {:3.0f}'.format(index, code, time, str(hour - 24).zfill(2), ws, wd)
            with open(out_file, 'a') as f:
                f.write(line + '\n')


def produce_fishzone(ini_time, shift_hours, fishcell_hour):
    # --检查数据是否全
    ec_ens_path = '/home/qxs/bma/data/ecmwf_ens/{}/'.format(ini_time.format('YYYYMMDDHH'))
    gefs_path = '/home/qxs/bma/data/gefs_fcst/{}/'.format(ini_time.format('YYYYMMDDHH'))
    bma_ws = '/home/qxs/bma/data/bma_result/{}/'.format(ini_time.format('YYYYMMDDHH'))
    wind_info = {}
    if len(glob.glob(bma_ws + 'ws_expect*.nc')) == len(shift_hours):
        wind_info['ws'] = 'bma'
        uv_name = 'ws_{}*nc'.format(ini_time.format('YYYYMMDDHH')).zfill(2)
        if len(glob.glob(ec_ens_path + uv_name)) == len(shift_hours):
            wind_info['wd'] = 'ec_ens'
        elif len(glob.glob(gefs_path + uv_name)) == len(shift_hours):
            wind_info['wd'] = 'gefs'
    # --输出渔区信息
    if sorted(wind_info.keys()) == sorted(['ws', 'wd']):
        outpath ='/home/qxs/bma/fish_zone/output/'
        out_file = outpath + 'FishCell_{}-{}-{}-{}_SeaWind.txt'.format(ini_time.format('YYYY'), ini_time.format('MM'), ini_time.format('DD'), fishcell_hour)
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        with open(out_file, 'a') as f:
            f.write('FISHERYCELL_SeaWind_V100' + '\n')
        info_file = r'/home/qxs/bma/fish_zone/yuqu-new-20170503.txt'
        info = pd.read_csv(info_file, sep='\s+', header=None, index_col=0)
        info.columns = ['code', 'lat', 'lon']
        for index, row in list(info.iterrows())[:]:
            code = row['code']
            lat = row['lat']
            lon = row['lon']
            cell = FishCell(ini_time, ec_ens_path, gefs_path, bma_ws)
            cell.extract(shift_hours, wind_info, index, code, lat, lon, out_file)


if __name__ == '__main__':
    # --路径参数
    outpath ='/home/qxs/bma/fish_zone/output/'
    # --时间参数
    ini_time = arrow.get('2019-04-14 12:00')  # 预报起始时间
    shift_hours = range(24, 96 + 6, 6)
    produce_fishzone(ini_time, shift_hours)
