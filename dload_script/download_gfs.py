# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import requests
import arrow
import os
import iris
import pandas as pd

class GFS(object):
    def __init__(self, host, data_path):
        self.host = host
        self.data_path = data_path
        self.header = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/56.0.2924.87 Safari/537.36'}

    def get_data(self, date_list, valid_time):
        for fcst_time in date_list[:]:
            download_path = self.data_path + fcst_time + '/'
            try:
                os.makedirs(download_path)
            except Exception:
                pass
            finally:
                ini_time = arrow.get(fcst_time, 'YYYYMMDDHH').shift(hours=-valid_time)
                url = self.host + ini_time.format('YYYYMM') + '/' + ini_time.format('YYYYMMDD') + '/'
                file_name = 'gfs_4_{}_{}00_0{}.grb2'.format(ini_time.format('YYYYMMDD'), ini_time.format('HH'), valid_time)
                url = url + file_name
                print(url)
                download_file = download_path + file_name
                if not os.path.exists(download_file):
                    r = requests.get(url, headers=self.header, stream=True, timeout=10)
                    with open(download_file, 'wb') as f:
                        for data in r.iter_content(chunk_size=512):
                            if data:
                                f.write(data)
                                f.flush()

    def composite_wind(self, date_list, valid_time):
        for fcst_time in date_list[:1]:
            gfs_path = self.data_path + fcst_time + '/'
            ini_time = arrow.get(fcst_time, 'YYYYMMDDHH').shift(hours=-valid_time)
            gfs_name = 'gfs_4_{}_{}00_0{}.grb2'.format(ini_time.format('YYYYMMDD'), ini_time.format('HH'), valid_time)
            gfs_file = gfs_path + gfs_name
            cube = iris.load(gfs_file)
            print(cube)




def main():
    host = r'https://nomads.ncdc.noaa.gov/data/gfs4/'
    data_path = '/home/qxs/bma/ec_tmp/data/' + 'gfs_fcst/'
    start = '20180502'
    end = '20180928'
    date_list = list(map(lambda x: arrow.get(x).format('YYYYMMDDHH'), pd.date_range(start, end, freq='12h', closed='left')))
    shift_hour = range(24, 73, 24)
    for valid_time in shift_hour[:1]:
        gfs  = GFS(host, data_path)
        # gfs.get_data(date_list, valid_time)
        gfs.composite_wind(date_list, valid_time)



if __name__ == '__main__':
    main()