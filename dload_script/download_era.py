# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com

from ecmwfapi import ECMWFDataServer
import os

area = {'china': "70/40/-10/180"}
code = {'u10': '165.128', 'v10': '166.128'}
time = "2008-01-01_2018-01-01"
download_name = 'uv_{}.nc'.format(time)
data_path = "/home/qxs/bma/"
download_file = data_path +  download_name
param = ''
vars = ['u10', 'v10']
for var in vars:
    param = param + str(code[var]) + '/'
try:
    os.makedirs(data_path + time + '/')
except OSError:
    pass
finally:
    if not os.path.exists(download_file):
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
            'area': area["china"],

            'param': param[:-1],
            'date': "{}/to/{}".format("20080101", "20180101"),

            'format': "netcdf",
            'target': download_file
        })