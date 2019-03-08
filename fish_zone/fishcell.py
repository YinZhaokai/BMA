# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import pandas as pd
import numpy as np
import xarray as xr
import glob
import arrow


def extract_data(ws_path, uv_path, ini_time, shift_hours):
    form = 'YYYYMMDDHH'
    info_file = r'/data1/Grid_201412-201602/detect/yuqu_div/yuqu-new-20170503.txt'
    info = pd.read_csv(info_file, sep='\s+', header=None, index_col=0)
    info.columns = ['code', 'lat', 'lon']
    for index, row in info.iterrows():
        code = row['code']
        lat = row['lat']
        lon = row['lon']
        locate = dict(longitude=lon, latitude=lat)
        for hour in shift_hours[:]:
            u_file = uv_path + ini_time + '/' + '*{}*u_control*'.format(arrow.get(ini_time, form).shift(hours=hour).format(form))
            v_file = uv_path + ini_time + '/' + '*{}*v_control*'.format(arrow.get(ini_time, form).shift(hours=hour).format(form))
            ws_file = ws_path + ini_time + '/' + 'ws_*{}.nc'.format(str(hour).zfill(2))
            time = np.datetime_as_string(xr.open_dataset(glob.glob(ws_file)[0])['time'].values[0], unit='s', timezone='UTC')
            time = arrow.get(time).to('Asia/Shanghai').format('YYYY-MM-DD HH:mm:ss')
            ws= xr.open_dataset(glob.glob(ws_file)[0])['wind_speed'].loc[locate].values[0]
            u = xr.open_dataset(glob.glob(u_file)[0])['u10'].loc[locate].values[0]
            v = xr.open_dataset(glob.glob(v_file)[0])['v10'].loc[locate].values[0]
            wd= 270. - 180/np.pi * np.arctan(v/u)
            line = '{:10d}, {}, {}, {}, {:4.1f}, {:3.0f}'.format(index, code, time, str(hour).zfill(2), ws, int(wd))
            print(line)


def main():
    # --路径参数
    ws_path = '/home/qxs/bma/data/bma_result/'
    uv_path = '/home/qxs/bma/data/ecmwf_fcst/'
    # --时间参数
    ini_time = '2018080712'   # 预报起始时间
    shift_hours = range(0, 72 + 12, 12)  # 预报时间间隔
    extract_data(ws_path, uv_path, ini_time, shift_hours)


if __name__ == '__main__':
    main()
