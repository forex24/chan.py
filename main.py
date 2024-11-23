from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver
import json

def dump_config(config: CChanConfig) -> str:
    """
    Dumps the config content as JSON
    
    Args:
        config: CChanConfig instance
    
    Returns:
        str: JSON formatted string of the config
    """
    def convert_value(v):
        """Helper function to convert values to JSON serializable format"""
        if hasattr(v, 'name'):  # Handle enum types
            return v.name
        elif isinstance(v, (str, int, float, bool, type(None))):
            return v
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, (list, tuple)):
            return [convert_value(item) for item in v]
        return str(v)  # Convert any other types to string

    config_dict = {
        "bi_conf": {
            "bi_algo": config.bi_conf.bi_algo,
            "is_strict": config.bi_conf.is_strict,
            "gap_as_kl": config.bi_conf.gap_as_kl,
            "bi_end_is_peak": config.bi_conf.bi_end_is_peak,
            "bi_allow_sub_peak": config.bi_conf.bi_allow_sub_peak,
            "bi_fx_check": convert_value(config.bi_conf.bi_fx_check)
        },
        "seg_conf": {
            "seg_algo": config.seg_conf.seg_algo,
            "left_method": convert_value(config.seg_conf.left_method)
        },
        "zs_conf": {k: convert_value(v) for k, v in vars(config.zs_conf).items()},
        "bs_point_conf": {
            "b_conf": {k: convert_value(v) for k, v in vars(config.bs_point_conf.b_conf).items()} if hasattr(config.bs_point_conf, 'b_conf') else {},
            "s_conf": {k: convert_value(v) for k, v in vars(config.bs_point_conf.s_conf).items()} if hasattr(config.bs_point_conf, 's_conf') else {}
        },
        "seg_bs_point_conf": {
            "b_conf": {k: convert_value(v) for k, v in vars(config.seg_bs_point_conf.b_conf).items()} if hasattr(config.seg_bs_point_conf, 'b_conf') else {},
            "s_conf": {k: convert_value(v) for k, v in vars(config.seg_bs_point_conf.s_conf).items()} if hasattr(config.seg_bs_point_conf, 's_conf') else {}
        },
    }
    
    # Remove any non-serializable objects
    for section_name, section in config_dict.items():
        if isinstance(section, dict):
            for key in list(section.keys()):
                if not isinstance(section[key], (str, int, float, bool, list, dict, type(None))):
                    del section[key]
    
    return json.dumps(config_dict, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    code = "sz.000001"
    begin_time = "2018-01-01"
    end_time = None
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig({
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


    json_str = dump_config(config)
    output_path = 'config_json.dump'
    with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(json_str), f, indent=4, ensure_ascii=False)