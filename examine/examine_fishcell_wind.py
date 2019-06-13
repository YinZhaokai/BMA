# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com


import arrow
import glob
import os
import xarray as xr
import geopandas
import salem
import numpy as np
import warnings
import sys
sys.path.append("/home/qxs/bma")
import wind_map


class NCData():
    def __init__(self, start, end, shift_hour):
        self.start = start
        self.end = end
        self.hour = shift_hour
        self.bma_path = '/home/qxs/bma/data/bma_result/'
        self.fnl_path = '/home/qxs/bma/data/fnl/'
        self.shp_path = '/home/qxs/bma/shp/fish_shp/'

    def merge(self, label, shift_hour):
        dataset = []
        time_list = []
        start = self.start
        end = self.end
        while start <= end:
            bma_path = self.bma_path + start.format('YYYYMMDDHH') + '/ws_expect_{}_{}.nc'.format(start.format('YYYYMMDDHH'), str(shift_hour).zfill(2))
            fnl_path = self.fnl_path + start.format('YYYYMMDDHH') + '/*_' + str(shift_hour).zfill(2) + '.nc'
            if glob.glob(bma_path) and glob.glob(fnl_path):
                time_list.append(start)
            start = start.shift(hours=12)
        for time in time_list:
            if label == 'bma':
                file_path = self.bma_path + time.format('YYYYMMDDHH') + '/ws_expect_{}_{}.nc'.format(time.format('YYYYMMDDHH'), str(shift_hour).zfill(2))
            else:
                file_path = self.fnl_path + time.format('YYYYMMDDHH') + '/*_' + str(shift_hour).zfill(2) + '.nc'
            try:
                file = glob.glob(file_path)[0]
            except Exception as e:
                print(file_path, e)
            else:
                data = xr.open_dataset(file)
                dataset.append(data)
        data_merge = xr.concat(dataset, dim='time')
        if label == 'bma':
            lat = data_merge.lat
            lon = data_merge.lon
        else:
            lat = data_merge.latitude
            lon = data_merge.longitude
        lon_range = lon[(lon > 100) & (lon < 132)]
        lat_range = lat[(lat > 0) & (lat < 43)]
        if label == 'bma':
            data_merge = data_merge.sel(lon=lon_range, lat=lat_range).drop('prob').rename({'lat': 'latitude', 'lon': 'longitude'})
        else:
            data_merge = data_merge.sel(longitude=lon_range, latitude=lat_range).squeeze('number').drop('heightAboveGround')
        return data_merge

    def cal_index(self):
        fishcell = geopandas.read_file(self.shp_path + 'FishCellALL.shp')
        bma_data = self.merge('bma', self.hour).ws
        fnl_data = self.merge('fnl', self.hour).ws
        # print(bma_data.sel(latitude=30, longitude=110))
        # print(fnl_data.sel(latitude=30, longitude=110))
        # print((abs(bma_data-fnl_data)).sel(latitude=30, longitude=110))
        # fnl_data = bma_data.copy(data=fnl_data.data[:, ::-1, :])
        rmse =  xr.Dataset({'ws': np.sqrt(((bma_data - fnl_data) ** 2).mean(dim='time'))}).rename({'longitude': 'lon', 'latitude':'lat'})
        mae = xr.Dataset({'ws': (abs(bma_data - fnl_data)).mean(dim='time')}).rename({'longitude': 'lon', 'latitude':'lat'})
        mre = xr.Dataset({'ws': (abs(bma_data - fnl_data) / fnl_data).mean(dim='time')}).rename({'longitude': 'lon', 'latitude':'lat'})
        # print(mae.ws.sel(lat=30, lon=110))
        new_lon = np.linspace(rmse.lon[0] - 0.25, rmse.lon[-1] - 0.25, rmse.lon.shape[0])
        new_lat = np.linspace(rmse.lat[0] - 0.25, rmse.lat[-1] - 0.25, rmse.lat.shape[0])
        rmse_cell = rmse.interp(lat=new_lat, lon=new_lon).salem.roi(shape=fishcell)
        mae_cell = mae.interp(lat=new_lat, lon=new_lon).salem.roi(shape=fishcell)
        mre_cell = mre.interp(lat=new_lat, lon=new_lon).salem.roi(shape=fishcell)
        rmse_region = {}
        mae_region = {}
        mre_region = {}
        seas = ['all', 'bohuang', 'dong', 'nan']
        seas_region = [{'latn': 43, 'lats': 0}, {'latn': 43, 'lats': 32}, {'latn': 32, 'lats': 22}, {'latn': 22, 'lats': 0}]
        for sea, region in list(zip(seas, seas_region)):
            rmse_region[sea] = rmse_cell.sel(lat=new_lat[(new_lat >= region['lats']) & (new_lat < region['latn'])]).ws.mean().values
            # index = np.where(np.where(np.isnan(rmse_cell.sel(lat=new_lat[(new_lat >= region['lats']) & (new_lat < region['latn'])]).ws), False, True) == True)
            # print(index[0].shape)
            mae_region[sea] = mae_cell.sel(lat=new_lat[(new_lat >= region['lats']) & (new_lat < region['latn'])]).mean().ws.values
            mre_region[sea] = mre_cell.sel(lat=new_lat[(new_lat >= region['lats']) & (new_lat < region['latn'])]).mean().ws.values
        print('rmse', rmse_region)
        print('mae', mae_region)
        print('mre', mre_region)
        return rmse, mae

    def plot(self):
        rmse, mae = self.cal_index()
        wind_map.WindNC(rmse).fishcell(np.arange(0, 2+0.2, 0.2), '{}h均方根误差'.format(str(self.hour)))
        wind_map.WindNC(mae).fishcell(np.arange(0, 2+0.2, 0.2), '{}h绝对误差'.format(str(self.hour)))


def main():
    start = arrow.get('2019-04-01')
    end = arrow.get('2019-05-30')
    interval = 24
    hours = range(24, 96 + interval, interval)  # 预报时间间隔
    for hour in hours[:]:
        print(hour)
        data = NCData(start, end, hour)
        # data.cal_index()
        data.plot()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()