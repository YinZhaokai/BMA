# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import pandas as pd
import numpy as np
import warnings
import arrow
import iris
import os
import cf_units as unit
from ecmwfapi import ECMWFDataServer

class ECFcst(object):
    def __init__(self, in_path, out_path, initial_time, valid_time):
        self.out_path = out_path
        self.uv_path = in_path + initial_time + '/'
        self.uv_name = arrow.get(initial_time, 'YYYYMMDDHH').shift(hours=valid_time).format('YYYYMMDDHH')
        self.uv_file = self.uv_path + self.uv_name
        self.ws_path = out_path + initial_time + '/'
        self.ws_name = initial_time + '_{}.nc'.format(str(valid_time))
        self.ws_file = self.ws_path + self.ws_name
        print('EC Forecast data processing: {}'.format(self.ws_name.split('.')[0]))

    def wind_composite(self):
        print('  EC forecast wind composite......')
        try:
            os.makedirs(self.ws_path)
        except OSError as e:
            pass
        finally:
            if not os.path.exists(self.ws_file):
                u_file = '{}*{}*u.nc'.format(self.uv_path, self.uv_name)
                v_file = '{}*{}*v.nc'.format(self.uv_path, self.uv_name)
                u_control_file = '{}*{}*u_control.nc'.format(self.uv_path, self.uv_name)
                v_control_file = '{}*{}*v_control.nc'.format(self.uv_path, self.uv_name)
                try:
                    u_cube = iris.load(u_file)[0][:, :, :, :]
                    v_cube = iris.load(v_file)[0][:, :, :, :]
                except Exception as e:
                    print(e)
                else:
                    ws = np.zeros(shape=(u_cube.shape[0], u_cube.shape[1] + 1, u_cube.shape[2], u_cube.shape[3]))
                    for member in  list(range(u_cube.shape[1] + 1))[:]:
                        if member != 0 :
                            constraint = iris.Constraint(ensemble_member = member)
                            u = u_cube.extract(constraint).data
                            v = v_cube.extract(constraint).data
                            ws[:, member, :, :] = (u**2 + v**2)**0.5
                        else:
                            u = iris.load(u_control_file)[0][:, :, :].data
                            v = iris.load(v_control_file)[0][:, :, :].data
                            ws[:, 0, :, :] = (u ** 2 + v ** 2) ** 0.5
                    ws_cube = iris.cube.Cube(ws, 'wind_speed', units='m s**-1')
                    ws_cube.add_dim_coord(u_cube.coords('time')[0], 0)
                    ws_cube.add_dim_coord(u_cube.coords('latitude')[0], 2)
                    ws_cube.add_dim_coord(u_cube.coords('longitude')[0], 3)
                    number = iris.coords.DimCoord(np.arange(u_cube.shape[1] + 1, dtype=np.int32),
                                                  standard_name=None,  long_name='ensemble_member', var_name='number')
                    ws_cube.add_dim_coord(number, 1)
                    iris.save(ws_cube, self.ws_file)

    def data_extract(self, date_list, valid_time, lat, lon, out_file):
        print('  EC forecast data extract......{}'.format(out_file))
        if os.path.exists(out_file):
            status = True
        else:
            status = False
        for initial_time in date_list:
            time_label = arrow.get(initial_time, 'YYYYMMDDHH').shift(hours=valid_time).format('YYYYMMDDHH')
            ws_name = initial_time + '_{}.nc'.format(str(valid_time))
            ws_file = self.out_path + initial_time + '/' + ws_name
            if not status:
                if os.path.exists(ws_file):
                    cubes = iris.load(ws_file)[0][0, :, :, :]
                    latlon_constraint = iris.Constraint(latitude=lat, longitude=lon)
                    ws = cubes.extract(latlon_constraint).data
                    with open(out_file, 'a') as f:
                        f.write(time_label + ' ')
                        f.write(' '.join(str(x) for x in ws))
                        f.write('\n')
                else:
                    print('    Not Found {}. So all value in it is 9999.'.format(ws_file))
                    with open(out_file, 'a') as f:
                        f.write(time_label + ' ')
                        f.write('9999. ' * 51)                #此处51为成员个数
                        f.write('\n')


class ECReanalysis(object):
    def __init__(self, tmp_path, start, end, valid_time=None):
        self.start = start
        self.end  = end
        self.valid_time = valid_time
        self.uv_name = 'uv_{}_{}.nc'.format(start, end)
        self.uv_file = tmp_path + self.uv_name
        self.ws_name = 'ws_{}_{}.nc'.format(start, end)
        self.ws_file = tmp_path + self.ws_name
        print('EC Reanalysis data porcessing......')

    def download(self, vars=['u10', 'v10'], region='china'):
        area = {'china': "70/40/-10/180"}
        code = {'u10': '165.128', 'v10': '166.128'}
        start = arrow.get(self.start, 'YYYYMMDD').format('YYYY-MM-DD')
        end = arrow.get(self.end, 'YYYYMMDD').format('YYYY-MM-DD')
        param = ''
        for var in vars:
            param = param + str(code[var]) + '/'
        if not os.path.exists(self.uv_file):
            server = ECMWFDataServer()
            server.retrieve({
                'class': "ei",
                'dataset': "interim",
                'stream': "oper",

                'levtype': "sfc",

                'type': "an",
                'time': "00/12",
                'step': "0",

                'grid': "0.5/0.5",
                'area': area[region],

                'param': param[:-1],
                'date': "{}/to/{}".format(start, end),

                'format': "netcdf",
                'target': self.uv_file
            })
        else:
            print('    {} already existed'.format(self.uv_file))

    def wind_composite(self):
        print('  EC reanalysis wind composite......')
        cmd = "cdo expr,'ws=sqrt(u10*u10+v10*v10)' {} {}".format(self.uv_file, self.ws_file)
        if not os.path.exists(self.ws_file):
            try:
                os.system(cmd)
            except OSError:
                raise
        else:
            print('    {} already existed'.format(self.ws_file))

    def data_extract(self, lat, lon, out_file):
        print('  EC reanalysis data extract......')
        if not os.path.exists(self.ws_file):
            print('    Reanalysis data not found')
        else:
            cubes = iris.load(self.ws_file)[0]
            start = arrow.get(self.start, 'YYYYMMDD').shift(hours=self.valid_time)
            end = arrow.get(self.end, 'YYYYMMDD').shift(days=-3, hours=12)
            latlon_constraint = iris.Constraint(latitude=lat, longitude=lon)
            t_constraint = iris.Constraint(time=lambda cell:
                                start.datetime.replace(tzinfo=None)<= cell.point <= end.datetime.replace(tzinfo=None))
            ws = cubes.extract(t_constraint & latlon_constraint).data
            time = start
            for value in ws:
                line = time.format('YYYYMMDDHH') + ' ' + str(value)
                time = time.shift(hours=12)
                with open(out_file, 'a') as f:
                    f.write(line)
                    f.write('\n')


def ec_fcst_process(ec_path, tmp_path, locates, start, end, valid_time):
    end = arrow.get(end, 'YYYYMMDD').shift(days=-3).format('YYYYMMDD')
    date_list = list(map(lambda x: arrow.get(x).format('YYYYMMDDHH'), pd.date_range(start, end, freq='12h', closed='left')))
    for initial_time in date_list[:]:
        fcst = ECFcst(ec_path, tmp_path, initial_time, valid_time)
        fcst.wind_composite()
    for locate in list(locates)[:]:
        out_file = tmp_path + 'fcst/' + 'fcst_{}_[{},{}]'.format(str(valid_time), str(locate[0]), str(locate[1]))
        fcst.data_extract(date_list, valid_time, locate[0], locate[1], out_file)

def ec_reanlys_process(tmp_path, locates, start, end, valid_time):
    reanlys = ECReanalysis(tmp_path, start, end, valid_time)
    reanlys.download()
    reanlys.wind_composite()
    for locate in locates:
        out_file = tmp_path + 'obs/' + 'obs_{}_[{},{}]'.format(str(valid_time), str(locate[0]), str(locate[1]))
        # reanlys.data_extract(lat, lon, out_file)
        if not os.path.exists(out_file):
            reanlys.data_extract(locate[0], locate[1], out_file)

def bma_methond(tmp_path, locates, valid_time):
    for locate in locates:
        fcst_file = tmp_path + 'fcst/' + 'fcst_{}_[{},{}]'.format(str(valid_time), str(locate[0]), str(locate[1]))
        obs_file = tmp_path + 'obs/' + 'obs_{}_[{},{}]'.format(str(valid_time), str(locate[0]), str(locate[1]))
        out_file = tmp_path + 'result/' + 'result_{}_[{},{}]'.format(str(valid_time), str(locate[0]), str(locate[1]))
        cmd = 'ifort module_bma.f90 main.f90 -o  bma_process && ./bma_process {} {} > {}'.format(fcst_file, obs_file, out_file)
        os.system(cmd)

def bma_composite(tmp_path, valid_time):
    t_unit = unit.Unit('hours since 2018-08-02 00:00:00', calendar='gregorian')
    time = range(0, 114 * 12, 12)
    lat_coord = iris.coords.DimCoord(np.arange(7, 42 + 1, 0.5)[::-1], standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord(np.arange(105, 130 + 1, 0.5), standard_name='longitude', units='degrees')
    time_coord = iris.coords.DimCoord(time, standard_name='time', units=t_unit)
    data = np.zeros(shape=(time_coord.shape[0], lat_coord.shape[0], lon_coord.shape[0]))
    for y, lat in enumerate(lat_coord.points):
        for x, lon in enumerate(lon_coord.points):
            bma_file = tmp_path + 'result/' + 'result_{}_[{},{}]'.format(valid_time, lat, lon)
            result = pd.read_table(bma_file, sep='\s+', header=None, na_values=[9999., -9999.],
                                   names=['time', 'expect', '0.1', '0.25', '0.5', '0.75', '0.95'])
            data[:, y, x] = result['expect'].values
    cube = iris.cube.Cube(data, 'wind_speed', units='m s**-1')
    cube.add_dim_coord(time_coord, 0)
    cube.add_dim_coord(lat_coord, 1)
    cube.add_dim_coord(lon_coord, 2)
    iris.save(cube, tmp_path + 'result_{}.nc'.format(valid_time))
    print(cube)

def data_process():
    ec_path = '/data2/ecmwf_dataset/wind_ensemble/'
    tmp_path = '/home/qxs/bma/ec_tmp/'
    time_range = ['20180501', '20180930']
    x = np.arange(105, 130 + 1, 0.5)
    y = np.arange(7, 42 + 1, 0.5)[::-1]
    lat, lon = np.meshgrid(y, x, indexing = 'ij')
    shift_hour = range(24, 73, 24)
    for valid_time in shift_hour[:1]:
        # ec_fcst_process(ec_path, tmp_path, zip(lat.flat[:], lon.flat[:]), time_range[0], time_range[1], valid_time)
        # ec_reanlys_process(tmp_path, zip(lat.flat[:], lon.flat[:]), time_range[0], time_range[1], valid_time)
        bma_methond(tmp_path, zip(lat.flat[:], lon.flat[:]), valid_time)
    # bma_composite(tmp_path, 24)

def main():
    warnings.filterwarnings('ignore')
    data_process()
    # draw()

if __name__ == '__main__':
    main()