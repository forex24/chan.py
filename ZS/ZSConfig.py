import json
from Common.ChanException import CChanException, ErrCode


class CZSConfig:
    def __init__(self, need_combine=True, zs_combine_mode="zs", one_bi_zs=False, zs_algo="normal"):
        self.need_combine = need_combine
        self.zs_combine_mode = zs_combine_mode
        self.one_bi_zs = one_bi_zs
        self.zs_algo = zs_algo
        self._validate()

    def _validate(self):
        """Validate configuration parameters"""
        if self.zs_combine_mode not in ["zs", "bi"]:
            raise CChanException(f"Invalid zs_combine_mode: {self.zs_combine_mode}", ErrCode.PARA_ERROR)
        if self.zs_algo not in ["normal", "strict"]:
            raise CChanException(f"Invalid zs_algo: {self.zs_algo}", ErrCode.PARA_ERROR)

    def to_json(self) -> str:
        """Convert config to JSON string"""
        return json.dumps({
            'need_combine': self.need_combine,
            'zs_combine_mode': self.zs_combine_mode,
            'one_bi_zs': self.one_bi_zs,
            'zs_algo': self.zs_algo
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'CZSConfig':
        """Create config from JSON string"""
        try:
            data = json.loads(json_str)
            return cls(
                need_combine=data.get('need_combine', True),
                zs_combine_mode=data.get('zs_combine_mode', 'zs'),
                one_bi_zs=data.get('one_bi_zs', False),
                zs_algo=data.get('zs_algo', 'normal')
            )
        except json.JSONDecodeError as e:
            raise CChanException(f"Invalid JSON format: {str(e)}", ErrCode.PARA_ERROR)
