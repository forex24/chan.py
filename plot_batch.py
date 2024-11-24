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
        "plot_kline": False,
        "plot_bi": False,
        "plot_seg": True,
        "plot_zs": False,
        "plot_bsp": False,
        "plot_segseg":True,
        "plot_segzs":True,
	"plot_segbsp":True,
    }
    plot_para = {
        "figure": {
            "x_range": 10000,
        },
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("batch_label.png")
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
        "min_zs_cnt": 1,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": False,
        "plot_bi": True,
        "plot_seg": True,
        "plot_segseg":True,
        "plot_eigen": False,
        "plot_zs": False,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    plot_para = {
        "seg": {
            # "plot_trendline": True,
        },
        "bi": {
            # "show_num": True,
            # "disp_end": True,
        },
        "figure": {
            "x_range": 100000,
        },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        }
    }


    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

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
