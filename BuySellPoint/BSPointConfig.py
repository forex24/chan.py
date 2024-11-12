import json
from typing import Dict, List, Optional

from Common.CEnum import BSP_TYPE, MACD_ALGO
from Common.func_util import _parse_inf
from Common.ChanException import CChanException, ErrCode


class CPointConfig:
    def __init__(self,
                 divergence_rate,
                 min_zs_cnt,
                 bsp1_only_multibi_zs,
                 max_bs2_rate,
                 macd_algo,
                 bs1_peak,
                 bs_type,
                 bsp2_follow_1,
                 bsp3_follow_1,
                 bsp3_peak,
                 bsp2s_follow_2,
                 max_bsp2s_lv,
                 strict_bsp3,
                 ):
        self.divergence_rate = divergence_rate
        self.min_zs_cnt = min_zs_cnt
        self.bsp1_only_multibi_zs = bsp1_only_multibi_zs
        self.max_bs2_rate = max_bs2_rate
        assert self.max_bs2_rate <= 1
        self.SetMacdAlgo(macd_algo)
        self.bs1_peak = bs1_peak
        self.tmp_target_types = bs_type
        self.target_types: List[BSP_TYPE] = []
        self.bsp2_follow_1 = bsp2_follow_1
        self.bsp3_follow_1 = bsp3_follow_1
        self.bsp3_peak = bsp3_peak
        self.bsp2s_follow_2 = bsp2s_follow_2
        self.max_bsp2s_lv: Optional[int] = max_bsp2s_lv
        self.strict_bsp3 = strict_bsp3
        self.parse_target_type()

    def parse_target_type(self):
        _d: Dict[str, BSP_TYPE] = {x.value: x for x in BSP_TYPE}
        if isinstance(self.tmp_target_types, str):
            self.tmp_target_types = [t.strip() for t in self.tmp_target_types.split(",")]
        for target_t in self.tmp_target_types:
            assert target_t in ['1', '2', '3a', '2s', '1p', '3b']
        self.target_types = [_d[_type] for _type in self.tmp_target_types]

    def SetMacdAlgo(self, macd_algo):
        _d = {
            "area": MACD_ALGO.AREA,
            "peak": MACD_ALGO.PEAK,
            "full_area": MACD_ALGO.FULL_AREA,
            "diff": MACD_ALGO.DIFF,
            "slope": MACD_ALGO.SLOPE,
            "amp": MACD_ALGO.AMP,
            "amount": MACD_ALGO.AMOUNT,
            "volumn": MACD_ALGO.VOLUMN,
            "amount_avg": MACD_ALGO.AMOUNT_AVG,
            "volumn_avg": MACD_ALGO.VOLUMN_AVG,
            "turnrate_avg": MACD_ALGO.AMOUNT_AVG,
            "rsi": MACD_ALGO.RSI,
        }
        self.macd_algo = _d[macd_algo]

    def set(self, k, v):
        v = _parse_inf(v)
        if k == "macd_algo":
            self.SetMacdAlgo(v)
        else:
            exec(f"self.{k} = {v}")

    def to_json(self) -> str:
        """Convert config to JSON string"""
        divergence_rate_str = "inf" if self.divergence_rate == float('inf') else str(self.divergence_rate)
        
        return json.dumps({
            'divergence_rate': divergence_rate_str,
            'min_zs_cnt': self.min_zs_cnt,
            'bsp1_only_multibi_zs': self.bsp1_only_multibi_zs,
            'max_bs2_rate': self.max_bs2_rate,
            'macd_algo': self.macd_algo.value,
            'bs1_peak': self.bs1_peak,
            'bs_type': self.tmp_target_types,
            'bsp2_follow_1': self.bsp2_follow_1,
            'bsp3_follow_1': self.bsp3_follow_1,
            'bsp3_peak': self.bsp3_peak,
            'bsp2s_follow_2': self.bsp2s_follow_2,
            'max_bsp2s_lv': self.max_bsp2s_lv,
            'strict_bsp3': self.strict_bsp3
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'CPointConfig':
        """Create config from JSON string"""
        try:
            data = json.loads(json_str)
            divergence_rate = float('inf') if data['divergence_rate'].lower() == 'inf' \
                else float(data['divergence_rate'])
            
            return cls(
                divergence_rate=divergence_rate,
                min_zs_cnt=data['min_zs_cnt'],
                bsp1_only_multibi_zs=data['bsp1_only_multibi_zs'],
                max_bs2_rate=data['max_bs2_rate'],
                macd_algo=data['macd_algo'],
                bs1_peak=data['bs1_peak'],
                bs_type=data['bs_type'],
                bsp2_follow_1=data['bsp2_follow_1'],
                bsp3_follow_1=data['bsp3_follow_1'],
                bsp3_peak=data['bsp3_peak'],
                bsp2s_follow_2=data['bsp2s_follow_2'],
                max_bsp2s_lv=data['max_bsp2s_lv'],
                strict_bsp3=data['strict_bsp3']
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise CChanException(f"Invalid JSON format or missing required fields: {str(e)}", ErrCode.PARA_ERROR)


class CBSPointConfig:
    def __init__(self, **args):
        self.b_conf = CPointConfig(**args)
        self.s_conf = CPointConfig(**args)

    def to_json(self) -> str:
        """Convert config to JSON string"""
        return json.dumps({
            'b_conf': json.loads(self.b_conf.to_json()),
            's_conf': json.loads(self.s_conf.to_json())
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'CBSPointConfig':
        """Create config from JSON string"""
        try:
            data = json.loads(json_str)
            instance = cls.__new__(cls)  # Create new instance without calling __init__
            instance.b_conf = CPointConfig.from_json(json.dumps(data['b_conf']))
            instance.s_conf = CPointConfig.from_json(json.dumps(data['s_conf']))
            return instance
        except (json.JSONDecodeError, KeyError) as e:
            raise CChanException(f"Invalid JSON format or missing required fields: {str(e)}", ErrCode.PARA_ERROR)

    def GetBSConfig(self, is_buy):
        return self.b_conf if is_buy else self.s_conf
