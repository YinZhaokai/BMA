# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import pandas as pd
import numpy as np
import xarray as xr
import arrow
import iris
import os
import glob
import requests
from pathos.pools import ProcessPool
from ecmwfapi import ECMWFDataServer
from tenacity import *
import paramiko


class NCData(object):
    def __init__(self, data_path, label=None):
        self.label = label
        self.data_path = data_path

    def merge_time(self, date_list):
        merge_name = 'ws_{}_{}-{}_{}.nc'.format(date_list[0]['ini'].format('YYYYMMDD'), str(date_list[0]['shift']).zfill(2),
                                                             date_list[-1]['ini'].format('YYYYMMDD'), str(date_list[-1]['shift']).zfill(2))
        merge_file = self.data_path + merge_name
        print('merge time: {}'.format(merge_file))
        for time in date_list[:]:
            ini_time = time['ini'].format('YYYYMMDDHH')
            shift_time = str(time['shift']).zfill(2)
            single_file = self.data_path + ini_time + '/' + 'ws_{}_{}.nc'.format(ini_time, shift_time)
            if os.path.exists(single_file):
                ws_exist = xr.open_dataset(single_file)
                break
        dataset = []
        for time in date_list[:]:
            ini_time = time['ini'].format('YYYYMMDDHH')
            shift_time = str(time['shift']).zfill(2)
            single_file = self.data_path + ini_time + '/' + 'ws_{}_{}.nc'.format(ini_time, shift_time)
            try:
                ws = xr.open_dataset(single_file)
            except IOError as e:
                if self.label == 'enforced':
                    ws = ws_exist.isel(time=0, drop=True)
                    ws_time = pd.to_datetime(time['ini'].shift(hours=time['shift']).format('YYYYMMDDHH'), format='%Y%m%d%H')
                    ws.coords['time']= ws_time
                    ws = ws.expand_dims('time', 0)
                    ws['ws'].values[:, :, :, :] = 9999.
                else:
                    print('{}: {}'.format(time, e))
                    merge_file = None
                    ens_num = 0
                    return merge_file, ens_num
            dataset.append(ws)
        ws_all = xr.auto_combine(dataset)
        for key in ws_all.dims.keys():
            if  key not in ['latitude', 'longitude', 'time', 'number']:
                ws_all = ws_all.squeeze(key)
        ens_num = ws_all['number'].shape[0]
        ws_all.to_netcdf(merge_file)
        return merge_file, ens_num

    def extract_all(self, merge_file, locate, time):
        extract_path = self.data_path + 'extract_{}/'.format(time.format('YYYYMMDDHH'))
        try:
            os.makedirs(extract_path)
        except OSError:
            pass
        finally:
            p = ProcessPool(16)
            for lat, lon in locate[:]:
                p.apipe(self.extract_point, merge_file, extract_path, lat, lon)
            p.close()
            p.join()
            p.clear()
            os.remove(merge_file)

    def extract_point(self, merge_file, extract_path, lat, lon):
        data = xr.open_dataarray(merge_file)
        time_list = data.coords['time'].values
        extract_name = '{}_[{},{}]'.format('point', str(lat), str(lon))
        extract_file = extract_path + extract_name
        # print('extract data: {} ---Run task {}'.format(extract_file, os.getpid()))
        for time in time_list[:]:
            ws = data.sel(latitude=lat, longitude=lon, time=time).values
            time = arrow.get(str(time)).format('YYYYMMDDHH')
            with open(extract_file, 'a') as f:
                f.write(time + ' ')
                if ws.shape:
                    f.write(' '.join(str(x) for x in ws))
                else:
                    f.write(' ' + str(ws))
                f.write('\n')


class ECReanalysis(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(ECReanalysis, self).__init__(self.data_path)

    @retry(stop=(stop_after_attempt(5)))
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
        uv = xr.open_dataset(uv_file)
        ws = xr.Dataset({'ws': (uv["u10"] ** 2 + uv["v10"] ** 2) ** 0.5}).expand_dims('number', 1)
        if not os.path.exists(composite_file):
            print('EC reanalysis wind composite: {}'.format(composite_file))
            ws.to_netcdf(composite_file)
        return composite_file


class ECFine(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.ec_path = '/data2/ecmwf_dataset/wind_Pac/{}/'.format(time['ini'].format('YYYY'))
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(ECFine, self).__init__(self.data_path)

    @retry(stop=(stop_after_attempt(5)))
    def download(self):
        time = self.time['ini'].shift(hours=self.time['shift']).format('YYYYMMDDHH')
        uv_file = 'xbt*{}*.nc'.format(time)
        download_path = self.data_path + self.base_path
        try:
            os.makedirs(download_path)
        except OSError:
            pass
        finally:
            uv_file = glob.glob(self.ec_path + uv_file)
            if uv_file:
                download_file = download_path +  uv_file[0].split('/')[-1]
                if not os.path.exists(download_file):
                    try:
                        os.symlink(uv_file[0],  download_file)
                    except OSError:
                        pass
            else:
                download_file = None
            return download_file

    def wind_composite(self, uv_file):
        ini_time = self.time['ini'].format('YYYYMMDDHH')
        shift_time = str(self.time['shift']).zfill(2)
        fcst_time = self.time['ini'].shift(hours=self.time['shift'])
        composite_name = 'ws_{}_{}.nc'.format(ini_time, shift_time)
        composite_file = self.data_path + self.base_path + composite_name
        try:
            uv = xr.open_dataset(uv_file).sel(time=fcst_time.datetime).expand_dims('time')
        except Exception as e:
            print('EC fine wind composite failed {}: {}'.format(e, composite_file))
        else:
            ws = xr.Dataset({'ws': (uv["u10"] ** 2 + uv["v10"] ** 2) ** 0.5}).expand_dims('number', 1)
            if not os.path.exists(composite_file):
                print('EC fine wind composite: {}'.format(composite_file))
                ws.to_netcdf(composite_file)


class ECEns(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.ec_path = '/data2/ecmwf_dataset/wind_ensemble/'
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(ECEns, self).__init__(self.data_path)

    @retry(stop=(stop_after_attempt(5)))
    def download(self):
        ini_time = self.time['ini'].format('YYYYMMDDHH') + '/'
        fcst_time = self.time['ini'].shift(hours=self.time['shift']).format('YYYYMMDDHH')
        u_file = '{}*{}*10u.nc'.format(ini_time, fcst_time)
        v_file = '{}*{}*10v.nc'.format(ini_time, fcst_time)
        u_control_file = '{}*{}*10u_control.nc'.format(ini_time, fcst_time)
        v_control_file = '{}*{}*10v_control.nc'.format(ini_time, fcst_time)
        uv_files = [u_file, v_file, u_control_file, v_control_file]
        download_path = self.data_path + self.base_path
        try:
            os.makedirs(download_path)
        except OSError:
            pass
        for uv_file in uv_files:
            uv_file = glob.glob(self.ec_path + uv_file)
            if uv_file:
                try:
                    os.symlink(uv_file[0],  download_path +  uv_file[0].split('/')[-1])
                except OSError:
                    pass
        download_files = {'u': glob.glob(download_path  + uv_files[0].split('/')[-1]),
                     'u_control': glob.glob(download_path  + uv_files[2].split('/')[-1]),
                     'v': glob.glob(download_path  + uv_files[1].split('/')[-1]),
                     'v_control': glob.glob(download_path  + uv_files[3].split('/')[-1])}
        return download_files

    def wind_composite(self, uv_file):
        ini_time = self.time['ini'].format('YYYYMMDDHH')
        shift_time = str(self.time['shift']).zfill(2)
        composite_name = 'ws_{}_{}.nc'.format(ini_time, shift_time)
        composite_file = self.data_path + self.base_path + composite_name
        try:
            os.makedirs(self.data_path + self.base_path)
        except OSError:
            pass
        finally:
            if not os.path.exists(composite_file):
                try:
                    u_cube = iris.load(uv_file['u'][0])[0][:, :, :, :]
                    v_cube = iris.load(uv_file['v'][0])[0][:, :, :, :]
                except IndexError:
                    print('EC Ens Wind Composite Failed: no uv files in {}'.format(self.data_path + self.base_path))
                else:
                    print('EC Ens wind composite: {}'.format(composite_file))
                    ws = np.zeros(shape=(u_cube.shape[0], u_cube.shape[1] + 1, u_cube.shape[2], u_cube.shape[3]))
                    for member in list(range(u_cube.shape[1] + 1))[:]:
                        if member != 0:
                            constraint = iris.Constraint(ensemble_member=member)
                            u = u_cube.extract(constraint).data
                            v = v_cube.extract(constraint).data
                            ws[:, member, :, :] = (u ** 2 + v ** 2) ** 0.5
                        else:
                            u = iris.load(uv_file['u_control'][0])[0][:, :, :].data
                            v = iris.load(uv_file['v_control'][0])[0][:, :, :].data
                            ws[:, 0, :, :] = (u ** 2 + v ** 2) ** 0.5
                    ws_cube = iris.cube.Cube(ws, 'wind_speed', units='m s**-1')
                    ws_cube.add_dim_coord(u_cube.coords('time')[0], 0)
                    ws_cube.add_dim_coord(u_cube.coords('latitude')[0], 2)
                    ws_cube.add_dim_coord(u_cube.coords('longitude')[0], 3)
                    number = iris.coords.DimCoord(np.arange(u_cube.shape[1] + 1, dtype=np.int32),
                                                  standard_name=None, long_name='number', var_name='number')
                    ws_cube.add_dim_coord(number, 1)
                    ws = xr.DataArray.from_iris(ws_cube).to_dataset().rename({'wind_speed': 'ws'})
                    ws.to_netcdf(composite_file)


class GFSFcst(object):
    def __init__(self):
        self.ip = '128.5.10.21'
        self.port = 22
        self.username = 'orca'
        self.password = 'chess@123'
        self.remote_path = '/public/home/orca/GFSDATA/'
        self.local_path = '/data2/gfs_dataset/'

    @retry(stop=(stop_after_attempt(5)))
    def download(self, time):
        time_path = time.format('YYYYMMDDHH') + '/'
        remote_file = self.remote_path + time_path + 'nwp_GFS_UV10_{}.nc'.format(time.format('YYYYMMDDHH'))
        local_file = self.local_path + time_path +  'nwp_GFS_UV10_{}.nc'.format(time.format('YYYYMMDDHH'))
        if not os.path.exists(local_file) or os.path.getsize(local_file) / float(1024 * 1024) < 210:
            try:
                os.mkdir(self.local_path + time_path)
            except Exception:
                pass
            ssh = paramiko.Transport((self.ip, self.port))
            ssh.connect(username=self.username, password=self.password)
            sftp = paramiko.SFTPClient.from_transport(ssh)
            try:
                sftp.get(remote_file, local_file)
            except Exception as e:
                print('{}: {}'.format(remote_file, e))
            finally:
                print('download {}'.format(local_file))
                sftp.close()
            if os.path.getsize(local_file) / float(1024 * 1024) < 210:
                os.remove(local_file)

    def wind_composite(self, uv_file):
        ini_time = self.time['ini'].format('YYYYMMDDHH')
        shift_time = str(self.time['shift']).zfill(2)
        composite_name = 'ws_{}_{}.nc'.format(ini_time, shift_time)
        composite_file = self.data_path + self.base_path + composite_name
        if not os.path.exists(composite_file):
            try:
                uv = xr.open_dataset(uv_file, engine='cfgrib',
                                          backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            except Exception as e:
                print('Wind Composite Failed: no uv files in {}: {}'.format(composite_file, e))
                return
            else:
                ws = xr.Dataset({'ws': (uv["u10"] ** 2 + uv["v10"] ** 2) ** 0.5})
                ws = ws.expand_dims('valid_time').drop(['time', 'step']).rename({'valid_time': 'time'}).expand_dims('number', 1)
                ws.to_netcdf(composite_file)
                print('GFS fcst wind composite: {}'.format(composite_file))
            os.system('rm {}'.format(self.data_path + self.base_path + '*.idx'))


class GEFSFcst(NCData):
    def __init__(self, data_path, time=None, base_path=None):
        self.gefs_path = '/data2/gefs_dataset/'
        self.host = r'https://www.ftp.ncep.noaa.gov/data/nccf/com/gens/prod/'
        self.header = {'User-Agent':
                           'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
        self.data_path = data_path
        self.time = time
        self.base_path = base_path + '/'
        super(GEFSFcst, self).__init__(self.data_path)

    @retry(stop=(stop_after_attempt(50)))
    def download(self, num):
        download_path = self.gefs_path + self.base_path
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
            if not os.path.exists(download_file) or os.path.getsize(download_file)/float(1024*1024) < 12:
                print('GEFS forecast download: {}'.format(download_file))
                response = requests.get(url, headers=self.header, stream=True, timeout=15)
                with open(download_file, 'wb') as f:
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
                try:
                    uv = xr.open_dataset(uv_file, engine='cfgrib',
                                         backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
                except Exception as e:
                    print('GEFS fcst Wind Composite Failed: no uv files {}: {}'.format(uv_file, e))
                    return
                else:
                    print('GEFS fcst wind composite: {}'.format(uv_file))
                    ws = xr.Dataset({'ws': (uv["u10"] ** 2 + uv["v10"] ** 2) ** 0.5})
                    ws = ws.expand_dims(['valid_time', 'number']).drop(['time', 'step']).rename({'valid_time': 'time'})
                    dataset.append(ws)
            ws_ens = xr.auto_combine(dataset)
            ws_ens.to_netcdf(composite_file)
        idx_files = glob.glob(self.data_path + self.base_path + '*.idx')
        for file in idx_files:
            os.remove(file)


class FNL(object):
    def __init__(self):
        self.ip = '128.5.6.18'
        self.port = 22
        self.username = 'qxs'
        self.password = 'qxs123'
        self.remote_path = '/share/wind/data/origin/NCEP0p5/'
        self.local_path = '/data2/fnl/'

    @retry(stop=(stop_after_attempt(5)))
    def download(self, time):
        mon_path = time.format('YYYYMM') + '/'
        remote_file = self.remote_path + mon_path + 'fnl_{}_{}_00'.format(time.format('YYMMDD'), time.format('HH'))
        local_file = self.local_path + mon_path + 'fnl_{}_00'.format(time.format('YYYYMMDDHH'))
        if not os.path.exists(local_file) or os.path.getsize(local_file) / float(1024 * 1024) < 120:
            try:
                os.mkdir(self.local_path + mon_path)
            except Exception:
                pass
            ssh = paramiko.Transport((self.ip, self.port))
            ssh.connect(username=self.username, password=self.password)
            sftp = paramiko.SFTPClient.from_transport(ssh)
            try:
                sftp.get(remote_file, local_file)
            except Exception as e:
                print('{}: {}'.format(remote_file, e))
            finally:
                print('download {}'.format(local_file))
                sftp.close()
            if os.path.getsize(local_file) / float(1024 * 1024) < 120:
                os.remove(local_file)
                # raise Exception('get {} failed, trying....'.format(local_file))
