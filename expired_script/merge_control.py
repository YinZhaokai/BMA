# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

import pandas as pd
import numpy as np
import arrow
import iris
import copy


class ECFcst(object):
    def __init__(self, in_path, out_path, time_list, valid_time):
        self.in_path = in_path
        self.out_path = out_path
        self.time_list = time_list
        self.valid_time = valid_time

    def composite(self):
        cube_list = iris.cube.CubeList()
        for fcst_time in self.time_list[:]:
            initial_time = arrow.get(fcst_time, 'YYYYMMDDHH').shift(hours=-self.valid_time).format('YYYYMMDDHH')
            uv_path = self.in_path + initial_time + '/'
            ufile = '{}*{}*u_control.nc'.format(uv_path, fcst_time)
            vfile = '{}*{}*v_control.nc'.format(uv_path, fcst_time)
            try:
                ucube = iris.load(ufile)[0]
                vcube = iris.load(vfile)[0]
            except Exception as e:
                print(e)
                cube = copy.deepcopy(cube_list[0])
                cube.data[:, :, :] = np.NAN
                first_time = cube.coords('time')[0].units.num2date(cube.coords('time')[0].points)[0]
                diff_hours = (arrow.get(fcst_time, 'YYYYMMDDHH') - arrow.get(first_time)).total_seconds() / 3600
                time = np.array(cube.coords('time')[0].points + diff_hours, dtype='int32')
                time_coord = iris.coords.DimCoord(time, standard_name='time',
                                                  long_name='time', var_name='time', units=cube.coords('time')[0].units)
                cube.remove_coord('time')
                cube.add_dim_coord(time_coord, 0)
            else:
                ucube.attributes.pop('history', None)
                vcube.attributes.pop('history', None)
                cube = (ucube ** 2 + vcube ** 2) ** 0.5
            finally:
                cube_list.append(cube)
        ws = cube_list.concatenate()[0]
        iris.save(ws, self.out_path + 'fcst_control_{}-{}_{}.nc'.format(self.time_list[0], self.time_list[-1], str(self.valid_time)))


def data_process():
    ec_path = '/data2/ecmwf_dataset/wind_ensemble/'
    tmp_path = '/home/qxs/bma/ec_tmp/'
    time_range = ['20180802', '20180928']
    start = arrow.get(time_range[0], 'YYYYMMDD').format('YYYYMMDD')
    end = arrow.get(time_range[1], 'YYYYMMDD').format('YYYYMMDD')
    time_list = list(map(lambda x: arrow.get(x).format('YYYYMMDDHH'), pd.date_range(start, end, freq='12h', closed='left')))
    shift_hour = range(24, 73, 24)
    for valid_time in shift_hour[:1]:
        fcst = ECFcst(ec_path, tmp_path, time_list, valid_time)
        fcst.composite()


def main():
    data_process()


if __name__ == '__main__':
    main()