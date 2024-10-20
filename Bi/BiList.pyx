# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, free
from typing import List, Optional, Union, overload

from Common.CEnum cimport FX_TYPE, KLINE_DIR
from KLine.KLine cimport CKLine

from .Bi cimport CBi
from .BiConfig cimport CBiConfig

cdef class CBiList:
    cdef:
        public list bi_list
        public CKLine last_end
        public CBiConfig config
        public list free_klc_lst

    def __cinit__(self, CBiConfig bi_conf=CBiConfig()):
        self.bi_list = []
        self.last_end = None  # 最后一笔的尾部
        self.config = bi_conf
        self.free_klc_lst = []  # 仅仅用作第一笔未画出来之前的缓存，为了获得更精准的结果而已，不加这块逻辑其实对后续计算没太大影响

    def __str__(self):
        return "\n".join([str(bi) for bi in self.bi_list])

    def __iter__(self):
        return iter(self.bi_list)

    @overload
    def __getitem__(self, index: int) -> CBi: ...

    @overload
    def __getitem__(self, index: slice) -> List[CBi]: ...

    def __getitem__(self, index: Union[slice, int]) -> Union[List[CBi], CBi]:
        return self.bi_list[index]

    def __len__(self):
        return len(self.bi_list)

    cpdef bint try_create_first_bi(self, CKLine klc):
        cdef CKLine exist_free_klc
        for exist_free_klc in self.free_klc_lst:
            if exist_free_klc.fx == klc.fx:
                continue
            if self.can_make_bi(klc, exist_free_klc):
                self.add_new_bi(exist_free_klc, klc)
                self.last_end = klc
                return True
        self.free_klc_lst.append(klc)
        self.last_end = klc
        return False

    cpdef bint update_bi(self, CKLine klc, CKLine last_klc, bint cal_virtual):
        # klc: 倒数第二根klc
        # last_klc: 倒数第1根klc
        cdef bint flag1 = self.update_bi_sure(klc)
        if cal_virtual:
            cdef bint flag2 = self.try_add_virtual_bi(last_klc)
            return flag1 or flag2
        else:
            return flag1

    cpdef bint can_update_peak(self, CKLine klc):
        if self.config.bi_allow_sub_peak or len(self.bi_list) < 2:
            return False
        if self.bi_list[-1].is_down() and klc.high < self.bi_list[-1].get_begin_val():
            return False
        if self.bi_list[-1].is_up() and klc.low > self.bi_list[-1].get_begin_val():
            return False
        if not end_is_peak(self.bi_list[-2].begin_klc, klc):
            return False
        if self[-1].is_down() and self[-1].get_end_val() < self[-2].get_begin_val():
            return False
        if self[-1].is_up() and self[-1].get_end_val() > self[-2].get_begin_val():
            return False
        return True

    cpdef bint update_peak(self, CKLine klc, bint for_virtual=False):
        if not self.can_update_peak(klc):
            return False
        cdef CBi _tmp_last_bi = self.bi_list[-1]
        self.bi_list.pop()
        if not self.try_update_end(klc, for_virtual=for_virtual):
            self.bi_list.append(_tmp_last_bi)
            return False
        else:
            if for_virtual:
                self.bi_list[-1].append_sure_end(_tmp_last_bi.end_klc)
            return True

    cpdef bint update_bi_sure(self, CKLine klc):
        # klc: 倒数第二根klc
        cdef CKLine _tmp_end = self.get_last_klu_of_last_bi()
        self.delete_virtual_bi()
        # 返回值：是否出现新笔
        if klc.fx == FX_TYPE.UNKNOWN:
            return _tmp_end != self.get_last_klu_of_last_bi()  # 虚笔是否有变
        if self.last_end is None or len(self.bi_list) == 0:
            return self.try_create_first_bi(klc)
        if klc.fx == self.last_end.fx:
            return self.try_update_end(klc)
        elif self.can_make_bi(klc, self.last_end):
            self.add_new_bi(self.last_end, klc)
            self.last_end = klc
            return True
        elif self.update_peak(klc):
            return True
        return _tmp_end != self.get_last_klu_of_last_bi()

    cpdef void delete_virtual_bi(self):
        if len(self) > 0 and not self.bi_list[-1].is_sure:
            cdef list sure_end_list = [klc for klc in self.bi_list[-1].sure_end]
            if len(sure_end_list):
                self.bi_list[-1].restore_from_virtual_end(sure_end_list[0])
                self.last_end = self[-1].end_klc
                for sure_end in sure_end_list[1:]:
                    self.add_new_bi(self.last_end, sure_end, is_sure=True)
                    self.last_end = self[-1].end_klc
            else:
                del self.bi_list[-1]
        self.last_end = self[-1].end_klc if len(self) > 0 else None
        if len(self) > 0:
            self[-1].next = None

    cpdef bint try_add_virtual_bi(self, CKLine klc, bint need_del_end=False):
        if need_del_end:
            self.delete_virtual_bi()
        if len(self) == 0:
            return False
        if klc.idx == self[-1].end_klc.idx:
            return False
        if (self[-1].is_up() and klc.high >= self[-1].end_klc.high) or (self[-1].is_down() and klc.low <= self[-1].end_klc.low):
            # ���新最后一笔
            self.bi_list[-1].update_virtual_end(klc)
            return True
        cdef CKLine _tmp_klc = klc
        while _tmp_klc and _tmp_klc.idx > self[-1].end_klc.idx:
            assert _tmp_klc is not None
            if self.can_make_bi(_tmp_klc, self[-1].end_klc, for_virtual=True):
                # 新增一笔
                self.add_new_bi(self.last_end, _tmp_klc, is_sure=False)
                return True
            elif self.update_peak(_tmp_klc, for_virtual=True):
                return True
            _tmp_klc = _tmp_klc.pre
        return False

    cpdef void add_new_bi(self, CKLine pre_klc, CKLine cur_klc, bint is_sure=True):
        cdef CBi new_bi = CBi(pre_klc, cur_klc, idx=len(self.bi_list), is_sure=is_sure)
        self.bi_list.append(new_bi)
        if len(self.bi_list) >= 2:
            self.bi_list[-2].next = self.bi_list[-1]
            self.bi_list[-1].pre = self.bi_list[-2]

    cpdef bint satisfy_bi_span(self, CKLine klc, CKLine last_end):
        # 检查两个K线组合之间是否满足笔的跨度要求
        # klc: 当前K线组合
        # last_end: 上一笔的结束K线组合

        # 获取两个K线组合之间的跨度
        cdef int bi_span = self.get_klc_span(klc, last_end)

        # 如果配置为严格模式，则跨度必须大于等于4
        if self.config.is_strict:
            return bi_span >= 4

        # 非严格模式下的处理
        cdef:
            int uint_kl_cnt = 0  # 统计包含的基本K线数量
            CKLine tmp_klc = last_end.next  # 从上一笔结束的下一个K线组合开始

        while tmp_klc:
            # 累加当前K线组合包含的基本K线数量
            uint_kl_cnt += len(tmp_klc.lst)

            # 处理最后一个虚笔的特殊情况
            if not tmp_klc.next:
                # 如果是最后一个K线组合，且klc紧接在last_end之后，则不满足笔的条件
                return False

            # 如果下一个K线组合的索引小于当前klc的索引，继续循环
            if tmp_klc.next.idx < klc.idx:
                tmp_klc = tmp_klc.next
            else:
                break

        # 非严格模式下，要求跨度大于等于3且包含的基本K线数量大于等于3
        return bi_span >= 3 and uint_kl_cnt >= 3

    cpdef int get_klc_span(self, CKLine klc, CKLine last_end):
        cdef:
            int span = klc.idx - last_end.idx
            CKLine tmp_klc = last_end
        if not self.config.gap_as_kl:
            return span
        if span >= 4:  # 加速运算，如果span需要真正精确的值，需要去掉这一行
            return span
        while tmp_klc and tmp_klc.idx < klc.idx:
            if tmp_klc.has_gap_with_next():
                span += 1
            tmp_klc = tmp_klc.next
        return span

    cpdef bint can_make_bi(self, CKLine klc, CKLine last_end, bint for_virtual=False):
        # 判断是否可以形成一笔
        
        # 1. 检查是否满足笔的跨度要求
        # 如果配置的笔算法为'fx'，则直接满足跨度要求
        # 否则，调用satisfy_bi_span方法检查是否满足跨度
        cdef bint satisify_span = True if self.config.bi_algo == 'fx' else self.satisfy_bi_span(klc, last_end)
        if not satisify_span:
            return False
        
        # 2. 检查分型是否有效
        # 使用last_end的check_fx_valid方法检查klc是否为有效的相反分型
        # 传入配置的bi_fx_check参数和for_virtual标志
        if not last_end.check_fx_valid(klc, self.config.bi_fx_check, for_virtual):
            return False
        
        # 3. 如果配置要求笔的端点必须是峰谷，则进行额外检查
        if self.config.bi_end_is_peak and not end_is_peak(last_end, klc):
            return False
        
        # 满足所有条件，可以形成一笔
        return True

    cpdef bint try_update_end(self, CKLine klc, bint for_virtual=False):
        if len(self.bi_list) == 0:
            return False
        cdef CBi last_bi = self.bi_list[-1]
        if (last_bi.is_up() and check_top(klc, for_virtual) and klc.high >= last_bi.get_end_val()) or \
           (last_bi.is_down() and check_bottom(klc, for_virtual) and klc.low <= last_bi.get_end_val()):
            last_bi.update_virtual_end(klc) if for_virtual else last_bi.update_new_end(klc)
            self.last_end = klc
            return True
        else:
            return False

    cpdef CKLine get_last_klu_of_last_bi(self):
        return self.bi_list[-1].get_end_klu() if len(self) > 0 else None

cdef bint check_top(CKLine klc, bint for_virtual):
    if for_virtual:
        return klc.dir == KLINE_DIR.UP
    else:
        return klc.fx == FX_TYPE.TOP

cdef bint check_bottom(CKLine klc, bint for_virtual):
    if for_virtual:
        return klc.dir == KLINE_DIR.DOWN
    else:
        return klc.fx == FX_TYPE.BOTTOM

cdef bint end_is_peak(CKLine last_end, CKLine cur_end):
    cdef:
        double cmp_thred
        CKLine klc
    if last_end.fx == FX_TYPE.BOTTOM:
        cmp_thred = cur_end.high  # 或者严格点选择get_klu_max_high()
        klc = last_end.get_next()
        while True:
            if klc.idx >= cur_end.idx:
                return True
            if klc.high > cmp_thred:
                return False
            klc = klc.get_next()
    elif last_end.fx == FX_TYPE.TOP:
        cmp_thred = cur_end.low  # 或者严格点选择get_klu_min_low()
        klc = last_end.get_next()
        while True:
            if klc.idx >= cur_end.idx:
                return True
            if klc.low < cmp_thred:
                return False
            klc = klc.get_next()
    return True
