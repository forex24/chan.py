from typing import List, Optional

from Bi.Bi import CBi
from Bi.BiList import CBiList
from Common.CEnum import BI_DIR, FX_TYPE, KLINE_DIR, SEG_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.func_util import revert_bi_dir

from .Eigen import CEigen


class CEigenFX:
    def __init__(self, _dir: BI_DIR, exclude_included=True, lv=SEG_TYPE.BI):
        self.lv = lv
        self.dir = _dir  # 线段方向
        self.ele: List[Optional[CEigen]] = [None, None, None]
        self.lst: List[CBi] = []
        self.exclude_included = exclude_included
        self.kl_dir = KLINE_DIR.UP if _dir == BI_DIR.UP else KLINE_DIR.DOWN
        self.last_evidence_bi: Optional[CBi] = None

    def add(self, bi: CBi) -> bool:  # 返回是否出现分形
        assert bi.dir != self.dir
        self.lst.append(bi)
        
        if self.ele[0] is None:
            self.ele[0] = CEigen(bi, self.kl_dir)
            return False
        
        if self.ele[1] is None:
            return self._add_second_element(bi)
        
        if self.ele[2] is None:
            return self._add_third_element(bi)
        
        raise CChanException(f"特征序列3个都找齐了还没处理!! 当前笔:{bi.idx},当前:{str(self)}", ErrCode.SEG_EIGEN_ERR)

    def _add_second_element(self, bi: CBi) -> bool:
        combine_dir = self.ele[0].try_add(bi, exclude_included=self.exclude_included)
        if combine_dir == KLINE_DIR.COMBINE:
            return False
        
        self.ele[1] = CEigen(bi, self.kl_dir)
        return not ((self.is_up() and self.ele[1].high < self.ele[0].high) or 
                    (self.is_down() and self.ele[1].low > self.ele[0].low))

    def _add_third_element(self, bi: CBi) -> bool:
        self.last_evidence_bi = bi
        allow_top_equal = (1 if bi.is_down() else -1) if self.exclude_included else None
        combine_dir = self.ele[1].try_add(bi, allow_top_equal=allow_top_equal)
        
        if combine_dir == KLINE_DIR.COMBINE:
            return False
        
        self.ele[2] = CEigen(bi, combine_dir)
        if not self._actual_break():
            return False
        
        self.ele[1].update_fx(self.ele[0], self.ele[2], exclude_included=self.exclude_included, allow_top_equal=allow_top_equal)
        fx = self.ele[1].fx
        return (self.is_up() and fx == FX_TYPE.TOP) or (self.is_down() and fx == FX_TYPE.BOTTOM)

    def reset(self):
        if self.exclude_included:
            bi_tmp_list = self.lst[1:]
            self.clear()
            return any(self.add(bi) for bi in bi_tmp_list)
        else:
            ele2_begin_idx = self.ele[1].lst[0].idx
            self.ele[0], self.ele[1], self.ele[2] = self.ele[1], self.ele[2], None
            self.lst = [bi for bi in self.lst[1:] if bi.idx >= ele2_begin_idx]
        return False

    def can_be_end(self, bi_lst: CBiList):
        if not self.ele[1].gap:
            return True
        
        end_bi_idx = self.GetPeakBiIdx()
        thred_value = bi_lst[end_bi_idx].get_end_val()
        break_thred = self.ele[0].low if self.is_up() else self.ele[0].high
        return self._find_revert_fx(bi_lst, end_bi_idx+2, thred_value, break_thred)

    def _find_revert_fx(self, bi_list: CBiList, begin_idx: int, thred_value: float, break_thred: float):
        first_bi_dir = bi_list[begin_idx].dir
        egien_fx = CEigenFX(revert_bi_dir(first_bi_dir), exclude_included=True, lv=self.lv)
        
        for bi in bi_list[begin_idx::2]:
            if egien_fx.add(bi):
                self.last_evidence_bi = bi
                return True
            
            if (bi.is_down() and bi._low() < thred_value) or (bi.is_up() and bi._high() > thred_value):
                return False
            
            if egien_fx.ele[1] is not None:
                if (bi.is_down() and egien_fx.ele[1].high > break_thred) or (bi.is_up() and egien_fx.ele[1].low < break_thred):
                    return True
        
        return None

    def _actual_break(self):
        if not self.exclude_included:
            return True
        
        if (self.is_up() and self.ele[2].low < self.ele[1][-1]._low()) or \
           (self.is_down() and self.ele[2].high > self.ele[1][-1]._high()):
            return True
        
        ele2_bi = self.ele[2][0]
        if ele2_bi.next and ele2_bi.next.next:
            if ele2_bi.is_down() and ele2_bi.next.next._low() < ele2_bi._low():
                self.last_evidence_bi = ele2_bi.next.next
                return True
            elif ele2_bi.is_up() and ele2_bi.next.next._high() > ele2_bi._high():
                self.last_evidence_bi = ele2_bi.next.next
                return True
        return False

    def is_down(self):
        return self.dir == BI_DIR.DOWN

    def is_up(self):
        return self.dir == BI_DIR.UP

    def GetPeakBiIdx(self):
        return self.ele[1].GetPeakBiIdx()

    def all_bi_is_sure(self):
        return all(bi.is_sure for bi in self.lst) and self.last_evidence_bi.is_sure

    def clear(self):
        self.ele = [None, None, None]
        self.lst = []

    def __str__(self):
        return " | ".join(f"{[] if ele is None else ','.join([str(b.idx) for b in ele.lst])}" for ele in self.ele)
