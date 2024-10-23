from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from Common.CEnum import DATA_FIELD
import pandas as pd
import os
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__ == "__main__":
    """
    用于把缠论元素保存到dataframe中
    需要把KLine_List里的to_dataframe细化
    """
    symbol = "eurusd"
    df = pd.read_csv(f'/opt/data/{symbol}.csv')
    print("csv readed")
    
    code = df
    begin_time = "2021-01-01"
    end_time = "2021-02-01"
    data_src = DATA_SRC.DATAFRAME
    lv_list = [KL_TYPE.K_1M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "divergence_rate": 0.8,
        "min_zs_cnt": 1,
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

    #is_hold = False
    #last_buy_price = None
    last_bsp_list = []
    for chan_snapshot in chan.step_load():  # 每增加一根K线，返回当前静态精算结果
        bsp_list = chan_snapshot.get_bsp()  # 获取买卖点列表
        if not bsp_list:  # 为空
            continue
        last_bsp = bsp_list[-1]  # 最后一个买卖点
        last_bsp_list.append(last_bsp)
        """
        print("last_bsp:", last_bsp)
        if BSP_TYPE.T1 not in last_bsp.type and BSP_TYPE.T1P not in last_bsp.type:  # 假如只做1类买卖点
            continue
        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
            continue
        if cur_lv_chan[-2].fx == FX_TYPE.BOTTOM and last_bsp.is_buy and not is_hold:  # 底分型形成后开仓
            last_buy_price = cur_lv_chan[-1][-1].close  # 开仓价格为最后一根K线close
            print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
            is_hold = True
        elif cur_lv_chan[-2].fx == FX_TYPE.TOP and not last_bsp.is_buy and is_hold:  # 顶分型形成后平仓
            sell_price = cur_lv_chan[-1][-1].close
            print(f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, profit rate = {(sell_price-last_buy_price)/last_buy_price*100:.2f}%')
            is_hold = False
        """
    
    last_bsp_df = pd.DataFrame([
                {
                    'bsp_type': bsp.type2str(),
                    'bi_idx': bsp.bi.idx if bsp.bi else None,
                    'time': bsp.klu.time,
                } for bsp in last_bsp_list
            ])
    directory = "test_chan"
    mkdir_p(directory)
    last_bsp_df.to_csv(os.path.join(directory, "last_bsp.csv"))
    chan[0].to_csv(directory)
