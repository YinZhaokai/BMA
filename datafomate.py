# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import random
import pandas as pd
import numpy as np
import xarray as xr
import arrow
import iris
import os
import glob
import copy
import requests
from pathos.pools import ProcessPool
import cf_units as unit
from ecmwfapi import ECMWFDataServer
from tenacity import *


class NCData(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.extract_path = data_path + 'extract/'

    def merge_time(self, date_list):
        merge_name = 'ws_{}_{}-{}_{}.nc'.format(date_list[0]['ini'].format('YYYYMMDD'), str(date_list[0]['shift']).zfill(2),
                                                             date_list[-1]['ini'].format('YYYYMMDD'), str(date_list[-1]['shift']).zfill(2))
        merge_file = self.data_path + merge_name
        try:
            os.remove(merge_file)
        except OSError:
            pass
        print('merge time: {}'.format(merge_file))
        cube_exsit_list = iris.cube.CubeList()
        for time in date_list[:]:
            ini_time = time['ini'].format('YYYYMMDDHH')
            shift_time = str(time['shift']).zfill(2)
            single_file = self.data_path + ini_time + '/' + 'ws_{}_{}.nc'.format(ini_time, shift_time)
            if os.path.exists(single_file):
                cube_exist = iris.load(single_file)[0]
                cube_exsit_list.append(cube_exist)
        cube_list = iris.cube.CubeList()
        for time in date_list[:]:
            ini_time = time['ini'].format('YYYYMMDDHH')
            shift_time = str(time['shift']).zfill(2)
            fcst_time = time['ini'].shift(hours=time['shift'])
            single_file = self.data_path + ini_time + '/' + 'ws_{}_{}.nc'.format(ini_time, shift_time)
            try:
                cube = iris.load(single_file)[0]
            except Exception as e:
                print('{}: {}'.format(time, e))
                cube = copy.deepcopy(cube_exsit_list[0])
                if len(cube.shape) == 4:
                    cube.data[0, :, :, :] = 9999.
                elif len(cube.shape) == 3:
                    cube.data[0, :, :] = 9999.
                first_time = cube.coords('time')[0].units.num2date(cube.coords('time')[0].points)[0]
                diff_hours = (fcst_time - arrow.get(first_time)).total_seconds() / 3600
                time = np.array(cube.coords('time')[0].points + diff_hours, dtype='int32')
                time_coord = iris.coords.DimCoord(time, standard_name=u'time',
                                                  long_name=u'time', var_name='time', units=cube.coords('time')[0].units)
                cube.remove_coord('time')
                cube.add_dim_coord(time_coord, 0)
            else:
                cube.attributes.pop('history', None)
            finally:
                cube_list.append(cube)
        ws = cube_list.concatenate()[0]
        iris.save(ws, merge_file)
        return merge_file

    def extract_all(self, merge_file, locate):
        try:
            os.makedirs(self.extract_path)
        except OSError:
            cmd = 'rm {}*'.format(self.extract_path)
            os.system(cmd)
        finally:
            p = ProcessPool(16)
            for lat, lon in list(locate)[:]:
                # self.extract_point(cube, prefix, lat, lon)
                p.apipe(self.extract_point, merge_file, lat, lon)
            p.close()
            p.join()
            p.clear()
            os.remove(merge_file)

    def extract_point(self, merge_file, lat, lon):
        data = xr.open_dataarray(merge_file)
        time_list = data.coords['time'].values
        # time_list = cube.coords('time')[0].points
        extract_name = '{}_[{},{}]'.format('point', str(lat), str(lon))
        extract_file = self.data_path + 'extract/' + extract_name
        if not os.path.exists(extract_file):
            print('extract data: {} ---Run task {}'.format(extract_file, os.getpid()))
            for num, time in enumerate(time_list[:]):
                ws = data.sel(latitude=lat, longitude=lon, time=time).values
                time = arrow.get(str(time)).format('YYYYMMDDHH')
                with open(extract_file, 'a') as f:
                    f.write(time + ' ')
                    if ws.shape:
                        f.write(' '.join(str(x) for x in ws))
                    else:
                        f.write(' ' + str(ws))
                    f.write('\n')


class ECFcst(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.ec_path = '/data2/ecmwf_dataset/wind_ensemble/'
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(ECFcst, self).__init__(self.data_path)

    def download(self):
        in_dir = self.time['ini'].format('YYYYMMDDHH') + '/'
        in_file = self.time['ini'].shift(hours=self.time['shift']).format('YYYYMMDDHH')
        u_file = '{}*{}*u.nc'.format(in_dir, in_file)
        v_file = '{}*{}*v.nc'.format(in_dir, in_file)
        u_control_file = '{}*{}*u_control.nc'.format(in_dir, in_file)
        v_control_file = '{}*{}*v_control.nc'.format(in_dir, in_file)
        uv_files = [u_file, v_file, u_control_file, v_control_file]
        out_path = self.data_path + self.base_path
        try:
            os.makedirs(out_path)
        except OSError:
            pass
        for uv_file in uv_files:
            uv_file = glob.glob(self.ec_path + uv_file)
            if uv_file:
                try:
                    os.symlink(uv_file[0],  out_path +  uv_file[0].split('/')[-1])
                except OSError:
                    pass
        out_files = {'u': glob.glob(out_path  + uv_files[0].split('/')[-1]),
                     'u_control': glob.glob(out_path  + uv_files[2].split('/')[-1]),
                     'v': glob.glob(out_path  + uv_files[1].split('/')[-1]),
                     'v_control': glob.glob(out_path  + uv_files[3].split('/')[-1])}
        return out_files

    def wind_composite(self, input_files):
        ini_time = self.time['ini'].format('YYYYMMDDHH')
        shift_time = str(self.time['shift']).zfill(2)
        # fcst_time = arrow.get(self.time['ini']).shift(hours=int(self.time['shift'])).format('YYYYMMDDHH')
        composite_name = 'ws_{}_{}.nc'.format(ini_time, shift_time)
        composite_file = self.data_path + self.base_path + composite_name
        try:
            os.makedirs(self.data_path + self.base_path)
        except OSError:
            pass
        finally:
            if not os.path.exists(composite_file):
                print('EC Fcst wind composite: {}'.format(composite_file))
                try:
                    u_cube = iris.load(input_files['u'][0])[0][:, :, :, :]
                    v_cube = iris.load(input_files['v'][0])[0][:, :, :, :]
                except Exception as e:
                    print('Wind Composite Failed: no uv files in {}'.format(self.data_path + self.base_path))
                else:
                    ws = np.zeros(shape=(u_cube.shape[0], u_cube.shape[1] + 1, u_cube.shape[2], u_cube.shape[3]))
                    for member in list(range(u_cube.shape[1] + 1))[:]:
                        if member != 0:
                            constraint = iris.Constraint(ensemble_member=member)
                            u = u_cube.extract(constraint).data
                            v = v_cube.extract(constraint).data
                            ws[:, member, :, :] = (u ** 2 + v ** 2) ** 0.5
                        else:
                            u = iris.load(input_files['u_control'][0])[0][:, :, :].data
                            v = iris.load(input_files['v_control'][0])[0][:, :, :].data
                            ws[:, 0, :, :] = (u ** 2 + v ** 2) ** 0.5
                    ws_cube = iris.cube.Cube(ws, 'wind_speed', units='m s**-1')
                    ws_cube.add_dim_coord(u_cube.coords('time')[0], 0)
                    ws_cube.add_dim_coord(u_cube.coords('latitude')[0], 2)
                    ws_cube.add_dim_coord(u_cube.coords('longitude')[0], 3)
                    number = iris.coords.DimCoord(np.arange(u_cube.shape[1] + 1, dtype=np.int32),
                                                  standard_name=None, long_name='ensemble_member', var_name='number')
                    ws_cube.add_dim_coord(number, 1)
                    iris.save(ws_cube, composite_file)
        # return composite_file


class ECReanalysis(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(ECReanalysis, self).__init__(self.data_path)

    @retry(stop=(stop_after_attempt(3)))
    def download(self, vars=['u10', 'v10'], region='china'):
        area = {'china': "70/40/-10/180"}
        code = {'u10': '165.128', 'v10': '166.128'}
        date = self.time['ini'].shift(hours=self.time['shift']).format('YYYY-MM-DD')
        hour = self.time['ini'].shift(hours=self.time['shift']).format('HH')
        download_name = 'uv_{}.nc'.format(self.time['ini'].format('YYYYMMDDHH'))
        download_file = self.data_path + self.base_path + download_name
        param = ''
        for var in vars:
            param = param + str(code[var]) + '/'
        try:
            os.makedirs(self.data_path + self.base_path)
        except OSError:
            pass
        finally:
            if not os.path.exists(download_file) or os.path.getsize(download_file)/float(1024) < 170:
                server = ECMWFDataServer()
                server.retrieve({
                    'class': "ei",
                    'dataset': "interim",
                    'stream': "oper",

                    'levtype': "sfc",

                    'type': "an",
                    'time': hour,
                    'step': "0",

                    'grid': "0.5/0.5",
                    'area': area[region],

                    'param': param[:-1],
                    'date': "{}/to/{}".format(date, date),

                    'format': "netcdf",
                    'target': download_file
                })
            return download_file

    def wind_composite(self, uv_file):
        ini_time = self.time['ini'].format('YYYYMMDDHH')
        shift_time = str(self.time['shift']).zfill(2)
        composite_name = 'ws_{}_{}.nc'.format(ini_time, shift_time)
        composite_file = self.data_path + self.base_path + composite_name
        cmd = "cdo expr,'ws=sqrt(u10*u10+v10*v10)' {} {}".format(uv_file, composite_file)
        if not os.path.exists(composite_file):
            print('EC reanalysis wind composite: {}'.format(composite_file))
            try:
                os.system(cmd)
            except OSError:
                raise
        return composite_file


class GFSFcst(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.host = r'https://nomads.ncdc.noaa.gov/data/gfs4/'
        self.header = {'User-Agent':
                           'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(GFSFcst, self).__init__(self.data_path)

    @retry(stop=(stop_after_attempt(3)))
    def download(self):
        download_path = self.data_path + self.base_path
        try:
            os.makedirs(download_path)
        except OSError:
            pass
        finally:
            url = self.host + self.time['ini'].format('YYYYMM') + '/' + self.time['ini'].format('YYYYMMDD') + '/'
            file_name = 'gfs_4_{}_{}00_{}.grb2'.format(self.time['ini'].format('YYYYMMDD'), self.time['ini'].format('HH'),
                                                       str(self.time['shift']).zfill(3))
            url = url + file_name
            download_file = download_path + file_name
            if not os.path.exists(download_file) or os.path.getsize(download_file)/float(1024*1024) < 54:
                print('GFS forecast download: {}'.format(download_file))
                response = requests.get(url, headers=self.header, stream=True, timeout=15)
                with open(download_file, 'ab') as f:
                    for data in response.iter_content(chunk_size=1024*1024):
                        if data:
                            f.write(data)
                            f.flush()
        return download_file

    def wind_composite(self, uv_file):
        ini_time = self.time['ini'].format('YYYYMMDDHH')
        shift_time = str(self.time['shift']).zfill(2)
        composite_name = 'ws_{}_{}.nc'.format(ini_time, shift_time)
        composite_file = self.data_path + self.base_path + composite_name
        # os.system('rm {}'.format(composite_file))
        if not os.path.exists(composite_file):
            uv = xr.open_dataset(uv_file, engine='cfgrib',
                                      backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            ws = xr.Dataset({'ws': (uv["u10"] ** 2 + uv["v10"] ** 2) ** 0.5}).expand_dims('valid_time').drop(['time', 'step'])
            ws = ws.rename({'valid_time': 'time'})
            ws.to_netcdf(composite_file)
            print('GFS fcst wind composite: {}'.format(composite_file))
            # cmd_nc =  'wgrib2 {} -netcdf {} -match \"(UGRD:10 m |VGRD:10 m )\" '.format(uv_file, tmp_file)
            # cmd_wind = 'cdo expr,\'ws=sqrt({}*{}+{}*{})\' {} {}'.format('u10', 'u10', 'v10', 'v10', tmp_file, composite_file)
            # os.system(cmd_nc)
            os.system('rm {}'.format(self.data_path + self.base_path + '*.idx'))
        return composite_file


class GEFSFcst(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.host = r'https://www.ftp.ncep.noaa.gov/data/nccf/com/gens/prod/'
        self.header = {'User-Agent':
                           'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(GEFSFcst, self).__init__(self.data_path)

    @retry(stop=(stop_after_attempt(5)))
    def download(self, num):
        download_path = self.data_path + self.base_path
        try:
            os.makedirs(download_path)
        except OSError:
            pass
        finally:
            date = self.time['ini'].format('YYYYMMDD')
            hour = self.time['ini'].format('HH')
            host = '{}gefs.{}/{}/pgrb2ap5/'.format(self.host, date, hour)
            if num == 0:
                file_name = 'gec00.t{}z.pgrb2a.0p50.f{}'.format(hour, str(self.time['shift']).zfill(3))
            else:
                file_name = 'gep{}.t{}z.pgrb2a.0p50.f{}'.format(str(num).zfill(2), hour, str(self.time['shift']).zfill(3))
            url = host + file_name
            download_file = download_path + file_name + '.grb2'
            if not os.path.exists(download_file) or os.path.getsize(download_file)/float(1024*1024) < 11:
                print('GEFS forecast download: {}'.format(download_file))
                response = requests.get(url, headers=self.header, stream=True, timeout=15)
                with open(download_file, 'ab') as f:
                    for data in response.iter_content(chunk_size=1024):
                        if data:
                            f.write(data)
                            f.flush()
        return download_file

    def wind_composite(self, uv_files):
        ini_time = self.time['ini'].format('YYYYMMDDHH')
        shift_time = str(self.time['shift']).zfill(2)
        composite_name = 'ws_{}_{}.nc'.format(ini_time, shift_time)
        composite_file = self.data_path + self.base_path + composite_name
        # os.system('rm {}'.format(composite_file))
        dataset = []
        if not os.path.exists(composite_file):
            for uv_file in uv_files:
                uv = xr.open_dataset(uv_file, engine='cfgrib',
                                     backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
                ws = xr.Dataset({'ws': (uv["u10"] ** 2 + uv["v10"] ** 2) ** 0.5}).expand_dims(['valid_time', 'number']).drop(['time', 'step'])
                ws = ws.rename({'valid_time': 'time'})
                dataset.append(ws)
            ws_ens = xr.auto_combine(dataset)
            ws_ens.to_netcdf(composite_file)
            print('GFS fcst wind composite: {}'.format(composite_file))
        os.system('rm {}'.format(self.data_path + self.base_path + '*.idx'))
        return composite_file


class BMA(object):
    def __init__(self, origin_path, locate):
        self.origin_path = origin_path
        self.extract_path = origin_path['bma_fcst'] + 'extract/'
        self.bma_point_path = origin_path['bma_fcst'] + 'bma_point/'
        self.locate = locate

    def process_all(self, train_num, all_num, ens_num1, ens_num2):
        dirs = [self.extract_path, self.bma_point_path]
        for dir_name in dirs:
            try:
                os.makedirs(dir_name)
            except OSError:
                pass
        p = ProcessPool(16)
        for lat, lon in list(self.locate)[:]:
            point_file = 'point_[{},{}]'.format(str(lat), str(lon))
            p.apipe(self.process_single, point_file, train_num, all_num, ens_num1, ens_num2)
        p.close()
        p.join()
        p.clear()
        for lat, lon in list(self.locate)[:]:
            point_file = 'point_[{},{}]'.format(str(lat), str(lon))
            if not os.path.exists(self.bma_point_path + point_file) or os.path.getsize(self.bma_point_path + point_file) < 790:
                self.process_single(point_file, train_num, all_num, ens_num1, ens_num2)
            # else:
            #     fsize = os.path.getsize(self.bma_point_path + point_file)
            #     if fsize < 790:
            #         self.process_single(point_file, train_num, all_num, ens_num1, ens_num2)

    def process_single(self, point_file, train_num, all_num, ec_num, gfs_num):
        ec_fcst_point_file = self.origin_path['ec_fcst']  + 'extract/' + point_file
        ec_reanlys_point_file = self.origin_path['ec_reanlys'] + 'extract/' + point_file
        gfs_fcst_point_file = self.origin_path['gfs_fcst'] + 'extract/' + point_file
        bma_point_file = self.bma_point_path + point_file
        fcst_cmd = 'join -a1 {} {} > {}'.format(ec_fcst_point_file, gfs_fcst_point_file, self.extract_path + point_file)
        os.system(fcst_cmd)
        print('BMA process: {} ---Run task {}'.format(bma_point_file, os.getpid()))
        cmd = r'ifort module_bma.f90 main.f90 -o bma_process && ./bma_process {} {} {} {} {} {} > {}'.format(
            self.extract_path + point_file, ec_reanlys_point_file, train_num, all_num, ec_num, gfs_num, bma_point_file)
        os.system(cmd)

    def point2grid(self, time_list):
        t_refernece = '1900-01-01 00:00:00'
        t_unit = unit.Unit('hours since {}'.format(t_refernece), calendar='gregorian')
        for n, time in enumerate(time_list):
            ini_time = time['ini'].format('YYYYMMDDHH')
            shift_time = str(time['shift']).zfill(2)
            fcst_time = time['ini'].shift(hours=time['shift'])
            delta_t = (fcst_time - arrow.get(t_refernece, 'YYYY-MM-DD HH:mm:ss')).total_seconds()/3600
            lat_coord = iris.coords.DimCoord(np.arange(self.locate[-1][0], self.locate[0][0] + 0.5, 0.5)[::-1],
                                             standard_name='latitude', units='degrees')
            lon_coord = iris.coords.DimCoord(np.arange(self.locate[0][1], self.locate[-1][1] + 0.5, 0.5),
                                             standard_name='longitude', units='degrees')
            time_coord = iris.coords.DimCoord(delta_t, standard_name='time', units=t_unit)
            data = np.zeros(shape=(time_coord.shape[0], lat_coord.shape[0], lon_coord.shape[0]))
            for y, lat in enumerate(lat_coord.points):
                for x, lon in enumerate(lon_coord.points):
                    bma_point_file = self.bma_point_path + 'point_[{},{}]'.format(lat, lon)
                    result = pd.read_table(bma_point_file, sep='\s+', header=None, na_values=[9999., -9999.],
                                           names=['time', 'expect', '0.1', '0.25', '0.5', '0.75', '0.95'])
                    data[:, y, x] = float(result[result.time==int(fcst_time.format('YYYYMMDDHH'))]['expect'])
            cube = iris.cube.Cube(data, 'wind_speed', units='m s**-1')
            cube.add_dim_coord(time_coord, 0)
            cube.add_dim_coord(lat_coord, 1)
            cube.add_dim_coord(lon_coord, 2)
            bma_grid_path = self.origin_path['bma_fcst'] + ini_time + '/'
            try:
                os.mkdir(bma_grid_path)
            except OSError:
                pass
            finally:
                iris.save(cube, bma_grid_path + 'ws_bma_{}_{}.nc'.format(ini_time, shift_time))
