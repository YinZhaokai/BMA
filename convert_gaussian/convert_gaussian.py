# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as mticker

def data_process(path, data_file):
    uv = xr.open_dataset(path + data_file)
    lc = uv.coords['longitude']
    la = uv.coords['latitude']
    region = dict(longitude=lc[(lc>100)&(lc<135)], latitude=la[(la>5)&(la<45)])
    ws = (uv["u10"].loc[region] ** 2 + uv["v10"].loc[region] ** 2) ** 0.5
    # --读取地形数据
    landsea = xr.open_dataset('/home/qxs/bma/data/landsea.nc')
    lc = landsea.coords['lon']
    la = landsea.coords['lat']
    region = dict(lon=lc[(lc>100) & (lc<135)], lat=la[(la>5) & (la<45)])
    landsea = landsea['LSMASK'].loc[region][::-1,:]
    # --地形数据插值
    new_lon = np.linspace(landsea.lon[0].values, landsea.lon[-1].values, landsea['lon'].shape[0] * 2 -1)
    new_lat = np.linspace(landsea.lat[0], landsea.lat[-1], landsea['lat'].shape[0] * 2 -1)
    landsea = landsea.interp(lat=new_lat, lon=new_lon)
    # --利用地形掩盖陆地数据
    ws.coords['mask'] = (('latitude', 'longitude'), landsea.values)
    ws = ws.where(ws.mask == 0)
    return ws

def draw(data):
    ocean_index = np.where(np.where(np.isnan(data.mean(dim='time')), False, True) == True)
    all_ws = []
    for ilat, ilon in list(zip(ocean_index[0], ocean_index[1]))[:]:
        # 此处加变换方法
        ws = data.isel(latitude=ilat, longitude=ilon)
        all_ws = np.append(all_ws, ws.values)
    all_ws = list(all_ws)
    # plot probability distribution
    sns.distplot(all_ws, kde=True)
    plt.show()
    # plot quantile-quantile
    fig = plt.figure()
    stats.probplot(all_ws, plot=plt)
    plt.show()

if __name__ == '__main__':
    data_path = "/home/qxs/bma/convert_gaussian/"
    data_file = "uv_2008-01-01_2018-01-01.nc"
    data = data_process(data_path, data_file)
    draw(data)