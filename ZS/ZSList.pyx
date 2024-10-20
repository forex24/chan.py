# cython: language_level=3
cimport cython
from typing import List, Union, overload

from Bi.Bi cimport CBi
from Bi.BiList cimport CBiList
from Common.func_util cimport revert_bi_dir
from Seg.Seg cimport CSeg
from Seg.SegListComm cimport CSegListComm
from .ZSConfig cimport CZSConfig
from .ZS cimport CZS

cdef class CZSList:
    cdef:
        public list zs_lst
        public CZSConfig config
        public list free_item_lst
        public int last_sure_pos

    def __cinit__(self, CZSConfig zs_config=CZSConfig()):
        self.zs_lst = []
        self.config = zs_config
        self.free_item_lst = []
        self.last_sure_pos = -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_last_pos(self, CSegListComm seg_list):
        self.last_sure_pos = -1
        cdef CSeg seg
        for seg in seg_list[::-1]:
            if seg.is_sure:
                self.last_sure_pos = seg.start_bi.idx
                return

    cpdef bint seg_need_cal(self, CSeg seg):
        return seg.start_bi.idx >= self.last_sure_pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void add_to_free_lst(self, object item, bint is_sure, str zs_algo):
        if len(self.free_item_lst) != 0 and item.idx == self.free_item_lst[-1].idx:
            # 防止笔新高或新低的更新带来bug
            self.free_item_lst = self.free_item_lst[:-1]
        self.free_item_lst.append(item)
        cdef CZS res = self.try_construct_zs(self.free_item_lst, is_sure, zs_algo)  # 可能是一笔中枢
        if res is not None and res.begin_bi.idx > 0:  # 禁止第一笔就是中枢的起点
            self.zs_lst.append(res)
            self.clear_free_lst()
            self.try_combine()

    cpdef void clear_free_lst(self):
        self.free_item_lst = []

    cpdef void update(self, CBi bi, bint is_sure=True):
        if len(self.free_item_lst) == 0 and self.try_add_to_end(bi):
            # zs_combine_mode=peak合并模式下会触发生效，=zs合并一定无效返回
            self.try_combine()  # 新形成的中枢尝试和之前的中枢合并
            return
        self.add_to_free_lst(bi, is_sure, "normal")

    cpdef bint try_add_to_end(self, CBi bi):
        return False if len(self.zs_lst) == 0 else self.zs_lst[-1].try_add_to_end(bi)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void add_zs_from_bi_range(self, list seg_bi_lst, BI_DIR seg_dir, bint seg_is_sure):
        cdef:
            int deal_bi_cnt = 0
            CBi bi
        for bi in seg_bi_lst:
            if bi.dir == seg_dir:
                continue
            if deal_bi_cnt < 1:  # 防止try_add_to_end执行到上一个线段的中枢里面去
                self.add_to_free_lst(bi, seg_is_sure, "normal")
                deal_bi_cnt += 1
            else:
                self.update(bi, seg_is_sure)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef CZS try_construct_zs(self, list lst, bint is_sure, str zs_algo):
        if zs_algo == "normal":
            if not self.config.one_bi_zs:
                if len(lst) == 1:
                    return None
                else:
                    lst = lst[-2:]
        elif zs_algo == "over_seg":
            if len(lst) < 3:
                return None
            lst = lst[-3:]
            if lst[0].dir == lst[0].parent_seg.dir:
                lst = lst[1:]
                return None
        cdef:
            double min_high = min(item._high() for item in lst)
            double max_low = max(item._low() for item in lst)
        return CZS(lst, is_sure=is_sure) if min_high > max_low else None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void cal_bi_zs(self, Union[CBiList, CSegListComm] bi_lst, CSegListComm seg_lst):
        while self.zs_lst and self.zs_lst[-1].begin_bi.idx >= self.last_sure_pos:
            self.zs_lst.pop()
        if self.config.zs_algo == "normal":
            cdef CSeg seg
            for seg in seg_lst:
                if not self.seg_need_cal(seg):
                    continue
                self.clear_free_lst()
                seg_bi_lst = bi_lst[seg.start_bi.idx:seg.end_bi.idx+1]
                self.add_zs_from_bi_range(seg_bi_lst, seg.dir, seg.is_sure)

            # 处理未生成新线段的部分
            if len(seg_lst):
                self.clear_free_lst()
                self.add_zs_from_bi_range(bi_lst[seg_lst[-1].end_bi.idx+1:], revert_bi_dir(seg_lst[-1].dir), False)
        elif self.config.zs_algo == "over_seg":
            assert self.config.one_bi_zs is False
            self.clear_free_lst()
            cdef int begin_bi_idx = self.zs_lst[-1].end_bi.idx+1 if self.zs_lst else 0
            cdef CBi bi
            for bi in bi_lst[begin_bi_idx:]:
                self.update_overseg_zs(bi)
        elif self.config.zs_algo == "auto":
            cdef:
                bint sure_seg_appear = False
                bint exist_sure_seg = seg_lst.exist_sure_seg()
                CSeg seg
                CBi bi
            for seg in seg_lst:
                if seg.is_sure:
                    sure_seg_appear = True
                if not self.seg_need_cal(seg):
                    continue
                if seg.is_sure or (not sure_seg_appear and exist_sure_seg):
                    self.clear_free_lst()
                    self.add_zs_from_bi_range(bi_lst[seg.start_bi.idx:seg.end_bi.idx+1], seg.dir, seg.is_sure)
                else:
                    self.clear_free_lst()
                    for bi in bi_lst[seg.start_bi.idx:]:
                        self.update_overseg_zs(bi)
                    break
        else:
            raise Exception(f"unknown zs_algo {self.config.zs_algo}")
        self.update_last_pos(seg_lst)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_overseg_zs(self, Union[CBi, CSeg] bi):
        if len(self.zs_lst) and len(self.free_item_lst) == 0:
            if bi.next is None:
                return
            if bi.idx - self.zs_lst[-1].end_bi.idx <= 1 and self.zs_lst[-1].in_range(bi.next) and self.zs_lst[-1].try_add_to_end(bi):
                return
        if len(self.zs_lst) and len(self.free_item_lst) == 0 and self.zs_lst[-1].in_range(bi) and bi.idx - self.zs_lst[-1].end_bi.idx <= 1:
            return
        self.add_to_free_lst(bi, bi.is_sure, zs_algo="over_seg")

    def __iter__(self):
        return iter(self.zs_lst)

    def __len__(self):
        return len(self.zs_lst)

    @overload
    def __getitem__(self, index: int) -> CZS: ...

    @overload
    def __getitem__(self, index: slice) -> List[CZS]: ...

    def __getitem__(self, Union[slice, int] index) -> Union[List[CZS], CZS]:
        return self.zs_lst[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void try_combine(self):
        if not self.config.need_combine:
            return
        while len(self.zs_lst) >= 2 and self.zs_lst[-2].combine(self.zs_lst[-1], combine_mode=self.config.zs_combine_mode):
            self.zs_lst = self.zs_lst[:-1]  # 合并后删除最后一个
