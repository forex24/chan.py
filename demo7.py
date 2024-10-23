import json
from typing import Dict, TypedDict

import xgboost as xgb

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
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
    }


if __name__ == "__main__":
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    code = "eurusd"
    train_begin_time = "2020-01-01"
    train_end_time = "2021-01-01"
    test_begin_time = "2021-07-01"
    test_end_time = "2021-12-31"
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

    # Training Data
    train_chan = CChan(
        code=code,
        begin_time=train_begin_time,
        end_time=train_end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    train_bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征
    bsp_count = 0
    # 跑策略，保存买卖点的特征
    for chan_snapshot in train_chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx not in train_bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第三元素出现时交易
            train_bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            train_bsp_dict[last_bsp.klu.idx]['feature'].add_feat(stragety_feature(last_klu))  # 开仓K线特征
            print(last_bsp.klu.time, last_bsp.is_buy)
            bsp_count = bsp_count + 1

    # print bsp_count
    print("bsp_count:", bsp_count)
    # 生成libsvm样本特征
    train_bsp_academy = [bsp.klu.idx for bsp in train_chan.get_bsp()]
    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    train_plot_marker = {}
    train_fid = open("train_feature.libsvm", "w")
    for bsp_klu_idx, feature_info in train_bsp_dict.items():
        label = int(bsp_klu_idx in train_bsp_academy)  # 以买卖点识别是否准确为label
        features = []  # List[(idx, value)]
        for feature_name, value in feature_info['feature'].items():
            if feature_name not in feature_meta:
                feature_meta[feature_name] = cur_feature_idx
                cur_feature_idx += 1
            features.append((feature_meta[feature_name], value))
        features.sort(key=lambda x: x[0])
        feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
        train_fid.write(f"{label} {feature_str}\n")
        train_plot_marker[feature_info["open_time"].to_str()] = ("√" if label else "×", "down" if feature_info["is_buy"] else "up")
    train_fid.close()

    with open("feature.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    # 画图检查label是否正确
    plot(train_chan, train_plot_marker)

    # load sample file & train model
    dtrain = xgb.DMatrix("train_feature.libsvm?format=libsvm")  # load sample
    param = {'max_depth': 2, 'eta': 0.3, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    evals_result = {}
    bst = xgb.train(
        param,
        dtrain=dtrain,
        num_boost_round=10,
        evals=[(dtrain, "train")],
        evals_result=evals_result,
        verbose_eval=True,
    )
    bst.save_model("model.json")

    # Testing Data
    test_chan = CChan(
        code=code,
        begin_time=test_begin_time,
        end_time=test_end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    test_bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

    # 跑策略，保存买卖点的特征
    for chan_snapshot in test_chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx not in test_bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第三元素出现时交易
            test_bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            test_bsp_dict[last_bsp.klu.idx]['feature'].add_feat(stragety_feature(last_klu))  # 开仓K线特征
            print(last_bsp.klu.time, last_bsp.is_buy)

    # 生成libsvm样本特征
    test_bsp_academy = [bsp.klu.idx for bsp in test_chan.get_bsp()]
    test_plot_marker = {}
    test_fid = open("test_feature.libsvm", "w")
    for bsp_klu_idx, feature_info in test_bsp_dict.items():
        label = int(bsp_klu_idx in test_bsp_academy)  # 以买卖点识别是否准确为label
        features = []  # List[(idx, value)]
        for feature_name, value in feature_info['feature'].items():
            features.append((feature_meta[feature_name], value))
        features.sort(key=lambda x: x[0])
        feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
        test_fid.write(f"{label} {feature_str}\n")
        test_plot_marker[feature_info["open_time"].to_str()] = ("√" if label else "×", "down" if feature_info["is_buy"] else "up")
    test_fid.close()

    # 画图检查label是否正确
    plot(test_chan, test_plot_marker)

    # load model
    model = xgb.Booster()
    model.load_model("model.json")

    # load test sample file
    dtest = xgb.DMatrix("test_feature.libsvm?format=libsvm")  # load sample

    # predict
    predictions = model.predict(dtest)
    print(predictions)
