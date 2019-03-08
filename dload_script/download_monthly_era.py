# -*- coding: utf-8 -*-
# author: GuoAnboyu
# email: guoappserver@gmail.com


from ecmwfapi import ECMWFDataServer
import pandas as pd
import arrow

def main():
    area = {'china': "70/40/-10/180", 'africa': "50/-30/-50/60"}
    code = {'u10': '165.128', 'v10': '166.128', 'cp': '143.128', 'tp': '228.128'}
    date = []
    start = arrow.get('1979/0101')
    end = arrow.get('2018/10/01')
    while start<end:
        date.append(start.format('YYYYMMDD'))
        start = arrow.get(start).shift(months=1)
    download_name = 'ERA-Interim_monthly_T_{}_{}.nc'.format(1979, 2018)
    download_file = r'D:\NMEFC_project\others\{}'.format(download_name)

    server = ECMWFDataServer()
    server.retrieve({
        'class': "ei",
        'dataset': "interim",
        'stream': "moda",

        'levtype': "sfc",
	"expver": "1",
        'type': "an",

        'grid': "1/1",

        'param': "167.128",
        'date': '/'.join(x for x in date),

        'format': "netcdf",
        'target': download_file
    })

if __name__ == '__main__':
    main()
