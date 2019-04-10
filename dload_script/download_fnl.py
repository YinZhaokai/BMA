# -*- coding: utf-8 -*-
# email: guoappserver@gmail.com

import arrow
import sys
sys.path.append("/home/qxs/bma/dload_script/")
from datadl_pack import FNL


def download_days():
    now_date = arrow.get().now().date()   # 预报起始时间
    # now = arrow.get().now().format('HH')
    now_list = [1, 7, 13, 19]
    for num in range(55)[::-1]:
        date = arrow.get(now_date).shift(days=-num)
        for now in now_list:
            if int(now) > 0 and int(now) < 6:
                ini_time = arrow.get(date).shift(days=-1, hours=12)
            elif int(now) > 6 and int(now) < 12:
                ini_time = arrow.get(date).shift(days=-1, hours=18)
            elif int(now) > 12 and int(now) < 18:
                ini_time = arrow.get(date)
            else:
                ini_time = arrow.get(date).shift(hours=6)
            fnl_file = FNL()
            fnl_file.download(ini_time)


def download_day():
    date = arrow.get().now().date()  # 预报起始时间
    now = arrow.get().now().format('HH')
    if int(now) > 0 and int(now) < 6:
        ini_time = arrow.get(date).shift(days=-1, hours=12)
    elif int(now) > 6 and int(now) < 12:
        ini_time = arrow.get(date).shift(days=-1, hours=18)
    elif int(now) > 12 and int(now) < 18:
        ini_time = arrow.get(date)
    else:
        ini_time = arrow.get(date).shift(hours=6)
    fnl_file = FNL()
    fnl_file.download(ini_time)


if __name__ == '__main__':
    download_day()