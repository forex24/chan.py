import argparse
import os
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver
import json
import pandas as pd

def load_csv(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot(chan):
    plot_config = {
        "plot_kline": True,
        "plot_bi": False,
        "plot_seg": True,
        "plot_zs": False,
        "plot_bsp": False,
        "plot_segseg":True,
        "plot_segzs":True,
    }
    plot_para = {
        "figure": {
            "x_range": 40000,
        },
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("no_trigger_label.png")
    #plot_driver.figure.show()

def main(df):
    code = df
    begin_time = None
    end_time = None
    data_src = DATA_SRC.DATAFRAME
    lv_list = [KL_TYPE.K_1M]

    config = CChanConfig({
        "trigger_step": False,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": False,
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

    #for _ in chan.step_load():
    #    pass

    plot(chan)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge CSV files for symbol analysis.')
    parser.add_argument('--root', default='/opt/data', help='Root directory for data (default: /opt/data)')
    parser.add_argument('--symbol', help='Specific symbol to process (optional)')
    
    args = parser.parse_args()

    root_directory = args.root
    input_directory = os.path.join(root_directory, 'raw_data')

    if args.symbol:
        csv_file = f"{args.symbol}.csv"
        file_path = os.path.join(input_directory, csv_file)
        df = load_csv(file_path)
        main(df)
