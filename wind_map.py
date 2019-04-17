# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import warnings
import glob
import arrow
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader, natural_earth
import geopandas
import salem
from matplotlib.colors import BoundaryNorm
import iris.plot as iplt
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class WindNC(object):
    def __init__(self, ds, path, ini_time, shift_hour):
        self.ds = ds.squeeze('time').drop(['time'])
        self.path = path
        self.ini_time = ini_time
        self.shift_hour = shift_hour

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
        shp_path = '/home/qxs/bma/shp/cn_shp/'
        # --创建画图空间
        proj = ccrs.PlateCarree()  # 创建坐标系
        fig = plt.figure(figsize=(6, 8), dpi=400)  # 创建页面
        ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  # 创建子图
        # --设置地图属性
        borders = cfeat.ShapelyFeature(Reader(shp_path + 'cntry02.shp').geometries(),
                                       proj, edgecolor='k', facecolor=cfeat.COLORS['land'])
        provinces = cfeat.ShapelyFeature(Reader(shp_path + 'bou2_4l.shp').geometries(),
                                       proj, edgecolor='k', facecolor=cfeat.COLORS['land'])
        # ax.add_feature(provinces, linewidth=0.6, zorder=2)
        ax.add_feature(borders, linewidth=0.6, zorder=10)
        ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)  # 加载分辨率为50的海岸线
        # ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)  # 加载分辨率为50的河流
        # ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)  # 加载分辨率为50的湖泊
        ax.set_extent([100, 131, 0, 42])
        # --设置网格点属性
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1.2, color='k', alpha=0.5, linestyle='--')
        gl.xlabels_top = False  # 关闭顶端的经纬度标签
        gl.ylabels_right = False  # 关闭右侧的经纬度标签
        gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
        gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
        gl.xlocator = mticker.FixedLocator(np.arange(95, 135+5, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(-5, 45+5, 5))
        return ax

    def power(self):
        levels = [0, 8., 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7, 70]
        levels_label = [u'<5级', u' 5级', u' 6级', u' 7级', u' 8级', u' 9级', u' 10级', u' 11级', u'>11级']
        wind = self.ds['ws'].to_iris()
        ax = self.map()
        # --设置颜色
        white = [1, 1, 1, 1]
        Spectral_r = cm.get_cmap('Spectral_r', len(levels_label))
        newcolors =Spectral_r(range(len(levels_label)))
        newcolors[0, : ] = white
        newcmp = ListedColormap(newcolors)
        norm = cm.colors.BoundaryNorm(levels, newcmp.N)
        # --画图
        im = iplt.contourf(wind, levels=levels, cmap=newcmp, axes=ax, norm=norm)
        # --画色标
        cbar = plt.colorbar(im, fraction=0.06, pad=0.04, norm=norm )
        cbar.set_ticks(list(map(lambda x: sum(levels[x: x + 2]) / 2, range(len(levels) - 1))))
        cbar.set_ticklabels(levels_label)
        # --标题
        fcst_time = self.ini_time.shift(hours=self.shift_hour).format('YYYY-MM-DD HH:mm')
        ini_time = self.ini_time.format('YYYY-MM-DD HH:mm')
        titile = '风力等级图 \n  预报时间:{} UTC  \n初始时间:{} UTC'.format(fcst_time, ini_time)
        ax.set_title(titile, fontsize=18)
        # --存图
        plt.savefig(self.path + 'ws_maxexpect_{}_{}.png'.format(self.ini_time.format('YYYYMMDDHH'), str(self.shift_hour)))
        # plt.show()

    def prob(self, level):
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
        try:
            plot = iplt.contourf(prob_da, levels=ticks, cmap='Spectral_r', axes=ax)
        except ValueError:
            print('{}级风概率图无数据，不出图。'.format(level))
        else:
            # --画色标
            cbar = plt.colorbar(plot, fraction=0.06, pad=0.04)
            cbar.set_ticks(list(map(lambda x: sum(ticks[x: x + 2]) / 2, range(len(ticks) - 1))))
            cbar.set_ticklabels(probs)
            # --标题
            fcst_time = self.ini_time.shift(hours=self.shift_hour).format('YYYY-MM-DD HH:mm')
            ini_time = self.ini_time.format('YYYY-MM-DD HH:mm')
            titile = '{}级风概率图 \n  预报时间:{} UTC  \n初始时间:{} UTC'.format(level, fcst_time, ini_time)
            ax.set_title(titile, fontsize=18)
            # --存图
            plt.savefig(self.path + 'ws_{}prob_{}_{}.png'.format(str(level), self.ini_time.format('YYYYMMDDHH'), str(self.shift_hour)))
            # plt.show()

    def fishcell(self):
        proj = ccrs.PlateCarree()
        levels = [0, 0.3, 1.6, 3.4, 5.5, 8., 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
        cmap = plt.get_cmap('Spectral_r')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        # --shp数据
        shp_path = '/home/qxs/bma/shp/fish_shp/'
        ax = self.map()
        fishcell = geopandas.read_file(shp_path + 'FishCellALL.shp')
        ax.add_geometries(fishcell.geometry, crs=proj, edgecolor='k', facecolor='none', zorder=9)
        # --风速数据
        wind = self.ds['ws']
        new_lon = np.linspace(wind.lon[0], wind.lon[-1], wind.lon.shape[0] * 2)
        new_lat = np.linspace(wind.lat[0], wind.lat[-1], wind.lat.shape[0]  * 2)
        wind = wind.interp(lat=new_lat, lon=new_lon)
        wind_mask = wind.salem.roi(shape=fishcell).to_iris()
        # --画图
        im = iplt.pcolormesh(wind_mask, cmap=cmap, axes=ax, norm=norm)
        # --画色标
        cbr = plt.colorbar(im, fraction=0.06, pad=0.04, norm=norm)
        cbr.set_ticks(levels)
        # --标题
        fcst_time = self.ini_time.shift(hours=self.shift_hour).format('YYYY-MM-DD HH:mm')
        ini_time = self.ini_time.format('YYYY-MM-DD HH:mm')
        titile = '渔区风力图 \n  预报时间:{} UTC  \n初始时间:{} UTC'.format(fcst_time, ini_time)
        ax.set_title(titile, fontsize=18)
        # --存图
        plt.savefig(self.path + 'ws_fishcell_{}_{}.png'.format(self.ini_time.format('YYYYMMDDHH'), str(self.shift_hour)))
        # plt.show()
    

def plot(path, ini_time, shift_hour):
    plot_types = ['expect', 'prob']
    for plot_type in plot_types[:]:
        files = glob.glob(path + '*{}*{}.nc'.format(plot_type, str(shift_hour)))
        ds = xr.open_dataset(files[0])
        wind = WindNC(ds, path, ini_time, shift_hour)
        if plot_type == 'expect':
            wind.power()
            wind.fishcell()
        elif plot_type == 'prob':
            pass
            wind.prob(6)
            wind.prob(8)


def plot_fishcell(path, ini_time, shift_hour):
    plot_types = ['expect']
    for plot_type in plot_types[:]:
        files = glob.glob(path + '*{}*{}.nc'.format(plot_type, str(shift_hour)))
        ds = xr.open_dataset(files[0])
        wind = WindNC(ds, path, ini_time, shift_hour)
        wind.fishcell()



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # --时间参数
    ini_time = arrow.get('2019040700', 'YYYYMMDDHH')
    # --路径参数
    path = '/home/qxs/bma/data/bma_result/{}/'.format(ini_time.format('YYYYMMDDHH'))
    shift_hours = range(24, 96 + 6, 6)  # 预报时间间隔
    for shift_hour in shift_hours[:1]:
        # plot(path, ini_time, shift_hour)
        plot_fishcell(path, ini_time, shift_hour)
