# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import glob
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader, natural_earth
import iris.plot as iplt
import matplotlib.pyplot as plt


class WindNC(object):
    def __init__(self, ds):
        self.ds = ds.squeeze('time').drop(['time'])

    def mask(self, label='land'):
        landsea = xr.open_dataset('/home/qxs/bma/data/landsea.nc')
        landsea = landsea['LSMASK']
        # --地形数据插值
        landsea = landsea.interp(lat=self.ds.lat.values, lon=self.ds.lon.values)
        # --利用地形掩盖陆地数据
        self.ds.coords['mask'] = (('lat', 'lon'), landsea.values)
        if label == 'land':
            self.ds = self.ds.where(self.ds.mask < 0.8)
        elif label == 'ocean':
            self.ds = self.ds.where(self.ds.mask > 0.2)
        self.ds = self.ds.squeeze('time').drop(['time', 'mask'])
        return self.ds

    def map(self):
        shp_path = '/home/qxs/bma/shp/'
        # --创建画图空间
        proj = ccrs.PlateCarree()  # 创建坐标系
        fig = plt.figure(figsize=(6, 8))  # 创建页面
        ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  # 创建子图
        # --设置地图属性
        borders = cfeat.ShapelyFeature(Reader(shp_path + 'cntry02.shp').geometries(),
                                       proj, edgecolor='k', facecolor=cfeat.COLORS['land'])
        provinces = cfeat.ShapelyFeature(Reader(shp_path + 'bou2_4l.shp').geometries(),
                                       proj, edgecolor='k', facecolor=cfeat.COLORS['land'])
        ax.add_feature(provinces, linewidth=0.6, zorder=2)
        ax.add_feature(borders, linewidth=1, zorder=1)
        ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)  # 加载分辨率为50的海岸线
        ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=1)  # 加载分辨率为50的河流
        ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=1)  # 加载分辨率为50的湖泊
        # --设置网格点属性
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1.2, color='k', alpha=0.5, linestyle='--')
        gl.xlabels_top = False  # 关闭顶端的经纬度标签
        gl.ylabels_right = False  # 关闭右侧的经纬度标签
        gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
        gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
        return ax

    def plot_wind_power(self):
        levels = [8., 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
        levels_label = [u'5级', u'6级', u'7级', u'8级', u'9级', u'10级', u'11级']
        wind = self.ds['ws'].to_iris()
        ax = self.map()
        # --画图
        plot = iplt.contourf(wind, levels=levels, cmap='Spectral_r', axes=ax, extend='both')
        plot.cmap.set_under('white')
        # --画色标
        cbar = plt.colorbar(plot)
        cbar.set_ticks(list(map(lambda x: sum(levels[x:x + 2]) / 2, range(len(levels) - 1))))
        cbar.set_ticklabels(levels_label)
        plt.show()

    def plot_wind_prob(self, level):
        levels = {6: [10.8, 13.9],
                    8: [17.2, 20.8]}
        ax = self.map()
        probs = list(map(lambda x: 1-float(x), self.ds.prob.values[::-1]))
        ticks = list(map(lambda x: x - 0.01, probs + [1]))
        wind_probs = np.full(self.ds['ws'].shape, np.nan)
        for n, prob in enumerate(probs):
            wind_value = self.ds['ws'].sel(prob=str(float('{:.2f}'.format(1- prob)))).values
            wind_prob = np.where((wind_value>=levels[level][0]) & (wind_value<levels[level][-1]), prob, np.nan)
            if n ==0:
                wind_probs[n, :, :] = wind_prob
            else:
                wind_probs[n, :, :] = np.where((wind_prob==prob), prob, wind_probs[n-1, :, :])
        prob_da = xr.DataArray(wind_probs[-1, :, :], coords=[self.ds.lat, self.ds.lon], dims=['lat', 'lon']).to_iris()
        # --画图
        plot = iplt.contourf(prob_da, levels=ticks, cmap='Spectral_r', axes=ax)
        # --画色标
        cbar = plt.colorbar(plot)
        cbar.set_ticks(list(map(lambda x: sum(ticks[x: x + 2]) / 2, range(len(ticks) - 1))))
        cbar.set_ticklabels(probs)
        plt.show()


if __name__ == '__main__':
    # --路径参数
    path = '/home/qxs/bma/data/bma_result/{}/'
    # --时间参数
    ini_time = '2019040100'
    shift_hours = range(24, 96 + 24, 24)  # 预报时间间隔
    # --数据类型
    plot_types = ['expect', 'prob']
    for shift_hour in shift_hours[:1]:
        for plot_type in plot_types[:]:
            files = glob.glob(path.format(ini_time) + '*{}*{}.nc'.format(plot_type, str(shift_hour)))
            ds = xr.open_dataset(files[0])
            wind = WindNC(ds)
            if plot_type == 'expect':
                wind.plot_wind_power()
            elif plot_type == 'prob':
                wind.plot_wind_prob(6)
