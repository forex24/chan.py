import json
from Common.CEnum import LEFT_SEG_METHOD
from Common.ChanException import CChanException, ErrCode


class CSegConfig:
    def __init__(self, seg_algo="chan", left_method="peak"):
        self.seg_algo = seg_algo
        if left_method == "all":
            self.left_method = LEFT_SEG_METHOD.ALL
        elif left_method == "peak":
            self.left_method = LEFT_SEG_METHOD.PEAK
        else:
            raise CChanException(f"unknown left_seg_method={left_method}", ErrCode.PARA_ERROR)

    def to_json(self) -> str:
        """Convert config to JSON string"""
        return json.dumps({
            'seg_algo': self.seg_algo,
            'left_method': self.left_method.name.lower()  # Convert enum to string
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'CSegConfig':
        """Create config from JSON string"""
        try:
            data = json.loads(json_str)
            return cls(
                seg_algo=data.get('seg_algo', 'chan'),
                left_method=data.get('left_method', 'peak')
            )
        except json.JSONDecodeError as e:
            raise CChanException(f"Invalid JSON format: {str(e)}", ErrCode.PARA_ERROR)
