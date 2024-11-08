import json
from Common.CEnum import FX_CHECK_METHOD
from Common.ChanException import CChanException, ErrCode


class CBiConfig:
    def __init__(
        self,
        bi_algo="normal",
        is_strict=True,
        bi_fx_check="half",
        gap_as_kl=True,
        bi_end_is_peak=True,
        bi_allow_sub_peak=True,
    ):
        self.bi_algo = bi_algo
        self.is_strict = is_strict
        if bi_fx_check == "strict":
            self.bi_fx_check = FX_CHECK_METHOD.STRICT
        elif bi_fx_check == "loss":
            self.bi_fx_check = FX_CHECK_METHOD.LOSS
        elif bi_fx_check == "half":
            self.bi_fx_check = FX_CHECK_METHOD.HALF
        elif bi_fx_check == 'totally':
            self.bi_fx_check = FX_CHECK_METHOD.TOTALLY
        else:
            raise CChanException(f"unknown bi_fx_check={bi_fx_check}", ErrCode.PARA_ERROR)

        self.gap_as_kl = gap_as_kl
        self.bi_end_is_peak = bi_end_is_peak
        self.bi_allow_sub_peak = bi_allow_sub_peak

    def to_json(self) -> str:
        """Convert config to JSON string"""
        return json.dumps({
            'bi_algo': self.bi_algo,
            'is_strict': self.is_strict,
            'bi_fx_check': self.bi_fx_check.name.lower(),  # Convert enum to string
            'gap_as_kl': self.gap_as_kl,
            'bi_end_is_peak': self.bi_end_is_peak,
            'bi_allow_sub_peak': self.bi_allow_sub_peak
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'CBiConfig':
        """Create config from JSON string"""
        try:
            data = json.loads(json_str)
            return cls(
                bi_algo=data.get('bi_algo', 'normal'),
                is_strict=data.get('is_strict', True),
                bi_fx_check=data.get('bi_fx_check', 'half'),
                gap_as_kl=data.get('gap_as_kl', True),
                bi_end_is_peak=data.get('bi_end_is_peak', True),
                bi_allow_sub_peak=data.get('bi_allow_sub_peak', True)
            )
        except json.JSONDecodeError as e:
            raise CChanException(f"Invalid JSON format: {str(e)}", ErrCode.PARA_ERROR)
