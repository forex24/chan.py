"""
20024-10-23 本文件主要目标：
1. 对品种做缠论形态分析，保存所有的信息，避免重复分析
2. 对品种做多周期分析，便于后续进一步分析
"""
import json
from typing import Dict, TypedDict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import sys
import os

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from pathlib import Path

def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime
    bsp_type: str


def chan_lab(base_path, symbol, start, end):
    """
    对输入数据打标策略产出的买卖点的特征
    start向前减一个月,end向后加一个月
    用于数据warmup
    """
    code = symbol
    begin_time = start - relativedelta(months=1)
    end_time = end + relativedelta(months=1)
    data_src = DATA_SRC.DF
    lv_list = [KL_TYPE.K_1M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 1,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
                "bsp_type": last_bsp.type2str()
            }
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(stragety_feature(last_klu))  # 开仓K线特征
            print(last_bsp.klu.time, last_bsp.is_buy, last_bsp.type2str())
