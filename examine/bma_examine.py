# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import glob
import os
import arrow
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
import numpy as np
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import iris.plot as iplt
from iris.analysis.geometry import geometry_area_weights
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader, natural_earth


class Pic(object):
    def __init__(self, path, bma_dir, reanlys_dir, fcst_dir, start, end):
        self.path = path
        self.bma_dir  = path + bma_dir
        self.reanlys_dir = path + reanlys_dir
        self.fcst_dir = path + fcst_dir
        self.start = arrow.get(start, 'YYYYMMDDHH')
        self.end = arrow.get(end, 'YYYYMMDDHH')

    def get_data(self):
        start = self.start
        end = self.end
        bma_cubes = iris.cube.CubeList()
        reanlys_cubes = iris.cube.CubeList()
        fcst_cubes = iris.cube.CubeList()
        while start <= end:
            fcst_file = glob.glob(self.fcst_dir + start.format('YYYYMMDDHH') + '/*')
            if fcst_file:
                bma_file = glob.glob(self.bma_dir + start.format('YYYYMMDDHH') + '/*')
                reanlys_file = glob.glob(self.reanlys_dir + start.format('YYYYMMDDHH') + '/ws*')
                bma_cubes.append(iris.load(bma_file)[0])
                reanlys_cubes.append(iris.load(reanlys_file)[0])
                fcst_cubes.append(iris.load(fcst_file)[0])
            start = start.shift(hours=12)
        bma_cubes = bma_cubes.concatenate_cube()
        equalise_attributes(reanlys_cubes)
        reanlys_cubes = reanlys_cubes.concatenate_cube()
        equalise_attributes(fcst_cubes)
        fcst_cubes = fcst_cubes.concatenate_cube()
        constraint = iris.Constraint(
            longitude = lambda cell: bma_cubes.coords('longitude')[0].points[0] <= cell <= bma_cubes.coords('longitude')[0].points[-1],
            latitude = lambda cell: bma_cubes.coords('latitude')[0].points[-1] <= cell <= bma_cubes.coords('latitude')[0].points[0])
        reanlys_cubes = reanlys_cubes.extract(constraint)
        reanlys_cubes.units = bma_cubes.units
        fcst_cubes = fcst_cubes.extract(constraint)
        fcst_cubes.units = bma_cubes.units

        bma_cubes.data = np.where(bma_cubes.data>100., np.nan, bma_cubes.data)
        bma_cubes.data = np.ma.masked_invalid(bma_cubes.data)
        reanlys_cubes.data = np.ma.masked_invalid(reanlys_cubes.data)
        fcst_cubes.data = np.ma.masked_invalid(fcst_cubes.data)
        fcst_cubes = fcst_cubes.collapsed('ensemble_member', iris.analysis.MEAN)

        bma_cubes.remove_coord('time')
        bma_cubes.remove_coord('latitude')
        bma_cubes.remove_coord('longitude')
        bma_cubes.add_dim_coord(reanlys_cubes.coords('time')[0], 0)
        bma_cubes.add_dim_coord(reanlys_cubes.coords('latitude')[0], 1)
        bma_cubes.add_dim_coord(reanlys_cubes.coords('longitude')[0], 2)

        # iris.save(bma_cubes, self.path + 'bma_cube.nc')
        # iris.save(reanlys_cubes, self.path + 'reanlys_cube.nc')

        landsea = iris.load(self.path + 'landsea.nc')[0][::-1,:]
        landsea = landsea.extract(constraint)
        landsea_regrid = landsea.regrid(bma_cubes, iris.analysis.Linear())
        return [bma_cubes, fcst_cubes, reanlys_cubes, landsea_regrid]

    def anom_map(self, title):
        # Read cube
        cube = self.get_data()
        landsea = cube[3]
        # print((cube[0][6:7,:,:]- cube[2][6:7:,:]).collapsed('time', iris.analysis.MEAN).data)
        cube = (cube[0] - cube[2])
        # iris.save(cube, self.path + 'bma-reanlys.nc')
        cube = cube.collapsed('time', iris.analysis.MEAN)
        # iris.save(cube, self.path + 'bma-reanlys_xy.nc')
        land_mask = np.where(landsea.data >= 0.75, True, False)
        # print(land_mask)

        cube_masked = cube.copy()
        cube_masked.data = np.ma.array(cube.data, mask=land_mask)
        # cube_masked.data = np.where(cube_masked.data > 100., 1, cube_masked.data)
        # iris.save(cube_masked, self.path + 'bma-control.nc')
        # constrain = iris.Constraint(latitude = lambda cell: cell>=25)
        # cube_masked = cube_masked.extract(constrain)
        print(np.nanmean(cube_masked.data))
        lon = cube.coords('longitude')[0]
        lat = cube.coords('latitude')[0]
        # Set up axes to show the map
        proj = ccrs.PlateCarree()
        fig =  plt.figure()
        ax = plt.axes(projection=proj)
        # Put a background image on for nice sea rendering.
        # ax.stock_img()
        #Set map feature
        borders = cfeat.ShapelyFeature(Reader(r'/home/qxs/bma/shp/cntry02.shp').geometries(),
                                       proj, edgecolor='k', facecolor=cfeat.COLORS['land'])
        provinces = cfeat.ShapelyFeature(Reader(r'/home/qxs/bma/shp/bou2_4l.shp').geometries(),
                                       proj, edgecolor='k', facecolor=cfeat.COLORS['land'])
        # ax.add_feature(provinces, zorder=1)
        ax.add_feature(borders, zorder=1)
        ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=1)
        ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=1)
        #Set Lat/Lon
        # ax.set_extent([105, 130, 7, 42])
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1.2, color='k', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 8, 'color': 'k', 'weight': 'normal'}
        gl.ylabel_style = {'size': 8, 'color': 'k', 'weight': 'normal'}
        gl.xlocator = mticker.FixedLocator(np.arange(lon.points[0], lon.points[-1] + 1, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(lat.points[-1], lat.points[0] + 1, 5))
        # normalization color from [0,1] and mapping onto the indices in the colormap.
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        # 设置levels
        levels = np.arange(-2, 2+0.2, 0.2)
        #Plot contour
        plot = iplt.contourf(cube_masked, axes=ax, levels=levels, cmap='RdBu_r', zorder=0)
        # iplt.contour(cube, levels=levels[::2], linewidths=0.5, colors='k', zorder=0)
        #Set colorbar
        plt.colorbar(plot, shrink=.8, orientation='vertical')
        plt.title(title)
        # plt.savefig(self.path + title + '.png')
        iplt.show()


def main():
    tmp_path = '/home/qxs/bma/ec_tmp/data/'
    reanlys_dir = 'ecmwf_reanlys/'
    fcst_dir = 'ecmwf_fcst/'
    bma_dir = 'bma_result/'
    start = '2018080800'
    end = '2018090612'
    plot = Pic(tmp_path, bma_dir, reanlys_dir, fcst_dir, start, end)
    # plot.get_data()
    plot.anom_map('BMA-Reanlysis')

if __name__ == '__main__':
    main()