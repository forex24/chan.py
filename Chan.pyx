# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, free
import copy
import datetime
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

from BuySellPoint.BS_Point cimport CBS_Point
from ChanConfig cimport CChanConfig
from Common.CEnum cimport AUTYPE, DATA_SRC, KL_TYPE
from Common.ChanException cimport CChanException, ErrCode
from Common.CTime cimport CTime
from Common.func_util cimport check_kltype_order, kltype_lte_day
from DataAPI.CommonStockAPI cimport CCommonStockApi
from KLine.KLine_List cimport CKLine_List
from KLine.KLine_Unit cimport CKLine_Unit

cdef class CChan:
    cdef:
        public str code
        public str begin_time
        public str end_time
        public AUTYPE autype
        public DATA_SRC data_src
        public list lv_list
        public CChanConfig conf
        public int kl_misalign_cnt
        public dict kl_inconsistent_detail
        public dict g_kl_iter
        public dict kl_datas
        public dict klu_cache
        public list klu_last_t

    def __cinit__(
        self,
        str code,
        begin_time=None,
        end_time=None,
        data_src: Union[DATA_SRC, str] = DATA_SRC.BAO_STOCK,
        list lv_list=None,
        CChanConfig config=None,
        AUTYPE autype = AUTYPE.QFQ,
    ):
        if lv_list is None:
            lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_60M]
        check_kltype_order(lv_list)  # lv_list顺序从高到低
        self.code = code
        self.begin_time = str(begin_time) if isinstance(begin_time, datetime.date) else begin_time
        self.end_time = str(end_time) if isinstance(end_time, datetime.date) else end_time
        self.autype = autype
        self.data_src = data_src
        self.lv_list = lv_list

        if config is None:
            config = CChanConfig()
        self.conf = config

        self.kl_misalign_cnt = 0
        self.kl_inconsistent_detail = defaultdict(list)

        self.g_kl_iter = defaultdict(list)

        self.do_init()

        if not config.trigger_step:
            for _ in self.load():
                pass

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj: CChan = cls.__new__(cls)
        memo[id(self)] = obj
        obj.code = self.code
        obj.begin_time = self.begin_time
        obj.end_time = self.end_time
        obj.autype = self.autype
        obj.data_src = self.data_src
        obj.lv_list = copy.deepcopy(self.lv_list, memo)
        obj.conf = copy.deepcopy(self.conf, memo)
        obj.kl_misalign_cnt = self.kl_misalign_cnt
        obj.kl_inconsistent_detail = copy.deepcopy(self.kl_inconsistent_detail, memo)
        obj.g_kl_iter = copy.deepcopy(self.g_kl_iter, memo)
        if hasattr(self, 'klu_cache'):
            obj.klu_cache = copy.deepcopy(self.klu_cache, memo)
        if hasattr(self, 'klu_last_t'):
            obj.klu_last_t = copy.deepcopy(self.klu_last_t, memo)
        obj.kl_datas = {}
        for kl_type, ckline in self.kl_datas.items():
            obj.kl_datas[kl_type] = copy.deepcopy(ckline, memo)
        for kl_type, ckline in self.kl_datas.items():
            for klc in ckline:
                for klu in klc.lst:
                    assert id(klu) in memo
                    if klu.sup_kl:
                        memo[id(klu)].sup_kl = memo[id(klu.sup_kl)]
                    memo[id(klu)].sub_kl_list = [memo[id(sub_kl)] for sub_kl in klu.sub_kl_list]
        return obj

    cpdef void do_init(self):
        self.kl_datas = {}
        for idx in range(len(self.lv_list)):
            self.kl_datas[self.lv_list[idx]] = CKLine_List(self.lv_list[idx], conf=self.conf)

    cpdef Iterable[CKLine_Unit] load_stock_data(self, CCommonStockApi stockapi_instance, KL_TYPE lv):
        cdef:
            int KLU_IDX
            CKLine_Unit klu
        for KLU_IDX, klu in enumerate(stockapi_instance.get_kl_data()):
            klu.set_idx(KLU_IDX)
            klu.kl_type = lv
            yield klu

    cpdef object get_load_stock_iter(self, object stockapi_cls, KL_TYPE lv):
        stockapi_instance = stockapi_cls(code=self.code, k_type=lv, begin_date=self.begin_time, end_date=self.end_time, autype=self.autype)
        return self.load_stock_data(stockapi_instance, lv)

    cpdef void add_lv_iter(self, object lv_idx, object iter):
        if isinstance(lv_idx, int):
            self.g_kl_iter[self.lv_list[lv_idx]].append(iter)
        else:
            self.g_kl_iter[lv_idx].append(iter)

    cpdef CKLine_Unit get_next_lv_klu(self, object lv_idx):
        if isinstance(lv_idx, int):
            lv_idx = self.lv_list[lv_idx]
        if len(self.g_kl_iter[lv_idx]) == 0:
            raise StopIteration
        try:
            return next(self.g_kl_iter[lv_idx][0])
        except StopIteration:
            self.g_kl_iter[lv_idx] = self.g_kl_iter[lv_idx][1:]
            if len(self.g_kl_iter[lv_idx]) != 0:
                return self.get_next_lv_klu(lv_idx)
            else:
                raise

    cpdef object step_load(self):
        assert self.conf.trigger_step

        self.do_init()

        yielded = False

        for idx, snapshot in enumerate(self.load(self.conf.trigger_step)):
            if idx < self.conf.skip_step:
                continue
            
            yield snapshot
            
            yielded = True

        if not yielded:
            yield self

    cpdef void trigger_load(self, dict inp):
        if not hasattr(self, 'klu_cache'):
            self.klu_cache = [None for _ in self.lv_list]

        if not hasattr(self, 'klu_last_t'):
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]

        for lv_idx, lv in enumerate(self.lv_list):
            if lv not in inp:
                if lv_idx == 0:
                    raise CChanException(f"最高级别{lv}没有传入数据", ErrCode.NO_DATA)
                continue

            for klu in inp[lv]:
                klu.kl_type = lv

            assert isinstance(inp[lv], list)

            self.add_lv_iter(lv, iter(inp[lv]))

        for _ in self.load_iterator(lv_idx=0, parent_klu=None, step=False):
            pass

        if not self.conf.trigger_step:
            for lv in self.lv_list:
                self.kl_datas[lv].cal_seg_and_zs()

    cpdef list init_lv_klu_iter(self, object stockapi_cls):
        lv_klu_iter = []
        valid_lv_list = []
        for lv in self.lv_list:
            try:
                lv_klu_iter.append(self.get_load_stock_iter(stockapi_cls, lv))
                valid_lv_list.append(lv)
            except CChanException as e:
                if e.errcode == ErrCode.SRC_DATA_NOT_FOUND and self.conf.auto_skip_illegal_sub_lv:
                    if self.conf.print_warning:
                        print(f"[WARNING-{self.code}]{lv}级别获取数据失败，跳过")
                    del self.kl_datas[lv]
                    continue
                raise e
        self.lv_list = valid_lv_list
        return lv_klu_iter

    cpdef object GetStockAPI(self):
        _dict = {}
        if self.data_src == DATA_SRC.BAO_STOCK:
            from DataAPI.BaoStockAPI import CBaoStock
            _dict[DATA_SRC.BAO_STOCK] = CBaoStock
        elif self.data_src == DATA_SRC.CCXT:
            from DataAPI.ccxt import CCXT
            _dict[DATA_SRC.CCXT] = CCXT
        elif self.data_src == DATA_SRC.CSV:
            from DataAPI.csvAPI import CSV_API
            _dict[DATA_SRC.CSV] = CSV_API
        elif self.data_src == DATA_SRC.DATAFRAME:
            from DataAPI.dataframeAPI import DATAFRAME_API
            _dict[DATA_SRC.DATAFRAME] = DATAFRAME_API
        if self.data_src in _dict:
            return _dict[self.data_src]
        assert isinstance(self.data_src, str)
        if self.data_src.find("custom:") < 0:
            raise CChanException("load src type error", ErrCode.SRC_DATA_TYPE_ERR)
        package_info = self.data_src.split(":")[1]
        package_name, cls_name = package_info.split(".")
        exec(f"from DataAPI.{package_name} import {cls_name}")
        return eval(cls_name)

    cpdef object load(self, bint step=False):
        stockapi_cls = self.GetStockAPI()
        try:
            stockapi_cls.do_init()
            for lv_idx, klu_iter in enumerate(self.init_lv_klu_iter(stockapi_cls)):
                self.add_lv_iter(lv_idx, klu_iter)
            self.klu_cache = [None for _ in self.lv_list]
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]

            yield from self.load_iterator(lv_idx=0, parent_klu=None, step=step)
            if not step:
                for lv in self.lv_list:
                    self.kl_datas[lv].cal_seg_and_zs()
        except Exception:
            raise
        finally:
            stockapi_cls.do_close()
        if len(self[0]) == 0:
            raise CChanException("最高级别没有获得任何数据", ErrCode.NO_DATA)

    cpdef void set_klu_parent_relation(self, CKLine_Unit parent_klu, CKLine_Unit kline_unit, KL_TYPE cur_lv, int lv_idx):
        if self.conf.kl_data_check and kltype_lte_day(cur_lv) and kltype_lte_day(self.lv_list[lv_idx-1]):
            self.check_kl_consitent(parent_klu, kline_unit)
        parent_klu.add_children(kline_unit)
        kline_unit.set_parent(parent_klu)

    cpdef void add_new_kl(self, KL_TYPE cur_lv, CKLine_Unit kline_unit):
        try:
            self.kl_datas[cur_lv].add_single_klu(kline_unit)
        except Exception:
            if self.conf.print_err_time:
                print(f"[ERROR-{self.code}]在计算{kline_unit.time}K线时发生错误!")
            raise

    cpdef void try_set_klu_idx(self, int lv_idx, CKLine_Unit kline_unit):
        if kline_unit.idx >= 0:
            return
        if len(self[lv_idx]) == 0:
            kline_unit.set_idx(0)
        else:
            kline_unit.set_idx(self[lv_idx][-1][-1].idx + 1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object load_iterator(self, int lv_idx, CKLine_Unit parent_klu, bint step):
        cdef:
            KL_TYPE cur_lv = self.lv_list[lv_idx]
            CKLine_Unit kline_unit
            bint is_last_lv = lv_idx == len(self.lv_list) - 1

        while True:
            try:
                kline_unit = self.get_next_lv_klu(lv_idx)
            except StopIteration:
                break

            if self.klu_cache[lv_idx] is not None:
                if kline_unit.time <= self.klu_cache[lv_idx].time:
                    continue
                self.add_new_kl(cur_lv, self.klu_cache[lv_idx])
                if parent_klu is not None:
                    self.set_klu_parent_relation(parent_klu, self.klu_cache[lv_idx], cur_lv, lv_idx)
                if is_last_lv and step:
                    yield self
                self.klu_cache[lv_idx] = None

            if kline_unit.time <= self.klu_last_t[lv_idx]:
                continue

            self.try_set_klu_idx(lv_idx, kline_unit)

            if not is_last_lv:
                yield from self.load_iterator(lv_idx + 1, kline_unit, step)

            if kline_unit.time > self.klu_last_t[lv_idx]:
                self.klu_last_t[lv_idx] = kline_unit.time
                self.klu_cache[lv_idx] = kline_unit

        if self.klu_cache[lv_idx] is not None:
            self.add_new_kl(cur_lv, self.klu_cache[lv_idx])
            if parent_klu is not None:
                self.set_klu_parent_relation(parent_klu, self.klu_cache[lv_idx], cur_lv, lv_idx)
            if is_last_lv and step:
                yield self
            self.klu_cache[lv_idx] = None

    cpdef void check_kl_consitent(self, CKLine_Unit parent_klu, CKLine_Unit kline_unit):
        cdef:
            double parent_open = parent_klu.open
            double parent_close = parent_klu.close
            double parent_high = parent_klu.high
            double parent_low = parent_klu.low
            double parent_amount = parent_klu.amount
            double parent_volume = parent_klu.volume
            double sub_open = kline_unit.open
            double sub_close = kline_unit.close
            double sub_high = kline_unit.high
            double sub_low = kline_unit.low
            double sub_amount = kline_unit.amount
            double sub_volume = kline_unit.volume

        if parent_open != sub_open or parent_close != sub_close or parent_high != sub_high or parent_low != sub_low:
            self.kl_misalign_cnt += 1
            self.kl_inconsistent_detail[parent_klu.time].append(
                f"parent: o={parent_open:.2f},c={parent_close:.2f},h={parent_high:.2f},l={parent_low:.2f},"
                f"a={parent_amount:.2f},v={parent_volume:.2f} "
                f"sub: o={sub_open:.2f},c={sub_close:.2f},h={sub_high:.2f},l={sub_low:.2f},"
                f"a={sub_amount:.2f},v={sub_volume:.2f}"
            )

    def __getitem__(self, idx):
        return self.kl_datas[self.lv_list[idx]]

    def __len__(self):
        return len(self.lv_list)
