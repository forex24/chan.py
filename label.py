"""
2024-10-18 最关键的文件,用于打标签
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

timeframe=['1d','8h','6h','4h','2h','1h','30min','15min','5min','3min']

# resample different timeframes
def resample_timeframe(df, timeframe):
    df_resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_resampled.dropna(inplace=True)
    return df_resampled


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

def CTime_to_datetime(ctime):
    return datetime(ctime.year, ctime.month, ctime.day, ctime.hour, ctime.minute, ctime.second)


def get_all_symbols(directory):
        # 查找目录下有多少个不同的 symbol
    symbols = set()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            symbol = filename.split('_')[0]
            symbols.add(symbol)
    return symbols

def check_signal_symbol_dup(symbol):
    # 初始化一个空的 DataFrame
    combined_df = pd.DataFrame()

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.startswith(f'{symbol}_feature') and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            # 将读取的 DataFrame 拼接到 combined_df 中
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 去重
    combined_df.drop_duplicates(inplace=True)
    combined_df = combined_df.sort_values(by='open_time')
    col = combined_df.pop('label')
    combined_df.insert(loc= 0 , column= 'label', value= col)

    # 检查 open_time 列中是否存在重复
    duplicates_in_open_time = combined_df[combined_df.duplicated('open_time', keep=False)]

    # 打印结果
    if duplicates_in_open_time.empty:
        print(f"No duplicates found in 'open_time' column for symbol {symbol}.")
    else:
        print(f"Duplicates found in 'open_time' column for symbol {symbol}:")
        print(duplicates_in_open_time)
        duplicates_in_open_time = duplicates_in_open_time.sort_values(by='open_time')

        # 将重复项写入 "{symbol}_all_dup.csv"
        dup_file_path = os.path.join(directory, f'{symbol}_all_dup.csv')
        duplicates_in_open_time.to_csv(dup_file_path, index=False)
        print(f"Duplicates for symbol {symbol} have been written to {dup_file_path}")

    # 将去重后的 DataFrame 写入 "{symbol}_all_feature.csv"
    directory = Path(directory).parent.absolute()
    output_file_path = os.path.join(directory, f'{symbol}_all_feature.csv')
    combined_df.to_csv(output_file_path, index=False)
    print(f"Combined DataFrame for symbol {symbol} has been written to {output_file_path}")


def check_dup_features(directory):
    # 查找目录下有多少个不同的 symbol
    symbols = get_all_symbols(directory)
    
    # 对每个 symbol 进行处理
    for symbol in symbols:
        check_signal_symbol_dup(symbol)


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

def label(base_path, symbol, start, end):
    try:
        chan_lab(base_path, symbol, start, end)
    except Exception as e:
        print(f"Error processing {symbol} from {start} to {end}: {e}")

def chan_lab(base_path, symbol, start, end):
    """
    对输入数据打标策略产出的买卖点的特征
    start向前减一个月,end向后加一个月
    用于数据warmup
    """
    code = symbol
    begin_time = start - relativedelta(months=1)
    end_time = end + relativedelta(months=1)
    data_src = DATA_SRC.CSV
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
    last_bsp_list=[]
    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if len(last_bsp_list) != 0:
            last = last_bsp_list[-1]
            if last is not None:
                if last_bsp.klu.time == last.klu.time:
                    continue
        last_bsp_list.append(last_bsp)

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

    last_bsp_df = pd.DataFrame([
                {
                    'is_buy': bsp.is_buy,
                    'bsp_type': bsp.type2str(),
                    'bi_idx': bsp.bi.idx if bsp.bi else None,
                    'time': bsp.klu.time,
                } for bsp in last_bsp_list
            ])
    directory = f"{symbol}_save_{start.strftime('%Y_%m_%d')}_{end.strftime('%Y_%m_%d')}"
    mkdir_p(directory)
    last_bsp_df.to_csv(os.path.join(directory, "last_bsp.csv"), index=False)
    chan[0].to_csv(directory)

    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    plot_marker = {}
    fname = f"{symbol}_feature_{start.strftime('%Y_%m_%d')}_{end.strftime('%Y_%m_%d')}"
    libsvm_filename = os.path.join(base_path, f"{fname}.libsvm")
    with open(libsvm_filename, "w") as fid:
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

        csv_filename = os.path.join(base_path, f"{fname}.csv")
        df_clear = df_all.drop(df[(df['open_time']<start) | (df['open_time']>end)].index)
        df_clear.to_csv(csv_filename, index=False)

    mata_filename = os.path.join(base_path, f"{fname}.meta")
    with open(mata_filename, "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    # 画图检查label是否正确
    # plot(chan, plot_marker)

def main(symbol, start_year):
    # 获取当前系统的cpu核心数
    n_cores = cpu_count()
    print(f'系统的核心数是：{n_cores}')


    base_path = 'features'
    base_path = os.path.join(os.getcwd(), base_path, symbol)
    mkdir_p(base_path)

    multi_work = Parallel(n_jobs=n_cores, backend='loky')
    tasks = []

    end_year = datetime.now().year
    for year in range(start_year, end_year + 1):
        start = datetime(year, 1, 1) #以年为单位划分任务
        end = start +  relativedelta(months=12)
        print('label from:' + str(start) + ' to:' + str(end))
        tasks.append(delayed(label)(base_path, symbol, start, end))

    res = multi_work(tasks)
    check_dup_features(base_path)

def main2():
    base_dir = os.path.join(os.getcwd(),'csv')
    symbols = get_all_symbols(base_dir)
    timeframe=['1d','8h','6h','4h','2h','1h','30min','15min','5min','3min']
    for freq in timeframe:
        for symbol in symbols:
            df = load_csv(os.path.join(base_dir, f'{symbol}.csv'))
            freq_df = resample_timeframe(df, freq)
            freq_df.to_csv(os.path.join(base_dir, f'{symbol}_{freq}.csv'))

# 这里要把预处理放到单独的文件里
def load_csv(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.sort_values(by='timestamp',inplace=True)
    return df

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
