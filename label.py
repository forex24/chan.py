import json
from typing import Dict, TypedDict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import sys

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver

class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime

def CTime_to_datetime(ctime):
    return datetime(ctime.year, ctime.month, ctime.day, ctime.hour, ctime.minute, ctime.second)

def plot(chan, plot_marker):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 400,
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("label.png")

def stragety_feature(last_klu):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
    }

def label(symbol, start, end):
    try:
        chan_lab(symbol, start, end)
    except Exception as e:
        print(f"Error processing {symbol} from {start} to {end}: {e}")

def chan_lab(symbol, start, end):
    """
    对输入数据打标策略产出的买卖点的特征
    """
    code = symbol
    begin_time = start
    end_time = end
    data_src = DATA_SRC.CSV
    lv_list = [KL_TYPE.K_1M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
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

    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    plot_marker = {}
    fname = f"feature_{start.strftime('%Y_%m_%d')}_{end.strftime('%Y_%m_%d')}"
    with open(f"{fname}.libsvm", "w") as fid:
        df_all = pd.DataFrame()
        for bsp_klu_idx, feature_info in bsp_dict.items():
            pd_features = {}
            label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
            pd_features['open_time'] = CTime_to_datetime(feature_info['open_time'])
            pd_features['bsp_type'] = feature_info['bsp_type']
            pd_features['is_buy'] = feature_info['is_buy']
            features = []  # List[(idx, value)]
            for feature_name, value in feature_info['feature'].items():
                if feature_name not in feature_meta:
                    feature_meta[feature_name] = cur_feature_idx
                    cur_feature_idx += 1
                features.append((feature_meta[feature_name], value))
                pd_features[feature_name] = value
            features.sort(key=lambda x: x[0])
            feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
            fid.write(f"{label} {feature_str}\n")
            pd_features['label'] = label
            df = pd.DataFrame(pd_features, index=[0])
            df_all = pd.concat([df_all, df], ignore_index=True)
            plot_marker[feature_info["open_time"].to_str()] = ("√" if label else "×", "down" if feature_info["is_buy"] else "up")

        df_all.to_csv(f"{fname}.csv", index=False)

    with open(f"{fname}.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    # 画图检查label是否正确
    # plot(chan, plot_marker)

def main(symbol, start_year):
    # 获取当前系统的cpu核心数
    n_cores = cpu_count()
    print(f'系统的核心数是：{n_cores}')

    multi_work = Parallel(n_jobs=n_cores, backend='loky')
    tasks = []

    end_year = datetime.now().year
    for year in range(start_year, end_year + 1):
        for quarter in range(0, 4):
            start = datetime(year, quarter * 3 + 1, 1)
            real_start = start - relativedelta(days=3)
            next_start = start + relativedelta(months=3)
            real_end = next_start + relativedelta(days=3)
            print('label from:' + str(real_start) + ' to:' + str(real_end))
            tasks.append(delayed(label)(symbol, real_start, real_end))

    res = multi_work(tasks)
    print(res)

if __name__ == "__main__":
    # 获取命令行参数，默认为 'eurusd'
    if len(sys.argv) > 1:
        symbol = sys.argv[1].lower()
    else:
        symbol = 'eurusd'

    # 获取起始年份，默认为 2000
    if len(sys.argv) > 2:
        start_year = int(sys.argv[2])
    else:
        start_year = 2000

    main(symbol, start_year)
