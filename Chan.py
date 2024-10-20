import copy
import datetime
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

from BuySellPoint.BS_Point import CBS_Point
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import check_kltype_order, kltype_lte_day
from DataAPI.CommonStockAPI import CCommonStockApi
from KLine.KLine_List import CKLine_List
from KLine.KLine_Unit import CKLine_Unit


class CChan:
    def __init__(
        self,
        code,
        begin_time=None,
        end_time=None,
        data_src: Union[DATA_SRC, str] = DATA_SRC.BAO_STOCK,
        lv_list=None,
        config=None,
        autype: AUTYPE = AUTYPE.QFQ,
    ):
        if lv_list is None:
            lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_60M]
        check_kltype_order(lv_list)  # lv_list顺序从高到低
        self.code = code
        self.begin_time = str(begin_time) if isinstance(begin_time, datetime.date) else begin_time
        self.end_time = str(end_time) if isinstance(end_time, datetime.date) else end_time
        self.autype = autype
        self.data_src = data_src
        self.lv_list: List[KL_TYPE] = lv_list

        if config is None:
            config = CChanConfig()
        self.conf = config

        self.kl_misalign_cnt = 0
        self.kl_inconsistent_detail = defaultdict(list)

        self.g_kl_iter = defaultdict(list)

        self.do_init()

        if not config.trigger_step:
            for _ in self.load():
                ...

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

    def do_init(self):
        self.kl_datas: Dict[KL_TYPE, CKLine_List] = {}
        for idx in range(len(self.lv_list)):
            self.kl_datas[self.lv_list[idx]] = CKLine_List(self.lv_list[idx], conf=self.conf)

    def load_stock_data(self, stockapi_instance: CCommonStockApi, lv) -> Iterable[CKLine_Unit]:
        for KLU_IDX, klu in enumerate(stockapi_instance.get_kl_data()):
            klu.set_idx(KLU_IDX)
            klu.kl_type = lv
            yield klu

    def get_load_stock_iter(self, stockapi_cls, lv):
        stockapi_instance = stockapi_cls(code=self.code, k_type=lv, begin_date=self.begin_time, end_date=self.end_time, autype=self.autype)
        return self.load_stock_data(stockapi_instance, lv)

    def add_lv_iter(self, lv_idx, iter):
        if isinstance(lv_idx, int):
            self.g_kl_iter[self.lv_list[lv_idx]].append(iter)
        else:
            self.g_kl_iter[lv_idx].append(iter)

    def get_next_lv_klu(self, lv_idx):
        if isinstance(lv_idx, int):
            lv_idx = self.lv_list[lv_idx]
        if len(self.g_kl_iter[lv_idx]) == 0:
            raise StopIteration
        try:
            return self.g_kl_iter[lv_idx][0].__next__()
        except StopIteration:
            self.g_kl_iter[lv_idx] = self.g_kl_iter[lv_idx][1:]
            if len(self.g_kl_iter[lv_idx]) != 0:
                return self.get_next_lv_klu(lv_idx)
            else:
                raise

    def step_load(self):
        # 步进加载方法，用于逐步加载和处理数据
        
        # 确保配置中的trigger_step为True，即启用了步进模式
        assert self.conf.trigger_step

        # 初始化数据，清空之前的数据以防止重复运行时出现问题
        self.do_init()

        # 标记是否曾经返回过结果
        yielded = False

        # 遍历load方法生成的每个数据快照
        for idx, snapshot in enumerate(self.load(self.conf.trigger_step)):
            # 如果当前索引小于配置的跳过步数，则继续下一次循环
            if idx < self.conf.skip_step:
                continue
            
            # 返回当前数据快照
            yield snapshot
            
            # 标记已经返回过结果
            yielded = True

        # 如果整个过程中没有返回过结果（可能是因为skip_step设置过大）
        # 则至少返回一次当前对象的状态
        if not yielded:
            yield self

    """
    整个数据加载过程的核心,包含了递归加载不同级别K线数据的逻辑。
    """
    def trigger_load(self, inp):
        # 触发加载新数据的方法
        # inp: 输入数据，格式为 {级别: [K线单元, ...]}

        # 初始化K线单元缓存，如果还没有的话
        if not hasattr(self, 'klu_cache'):
            self.klu_cache: List[Optional[CKLine_Unit]] = [None for _ in self.lv_list]

        # 初始化每个级别的最后时间，如果还没有的话
        if not hasattr(self, 'klu_last_t'):
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]

        # 遍历所有级别
        for lv_idx, lv in enumerate(self.lv_list):
            if lv not in inp:
                # 如果最高级别没有数据，抛出异常
                if lv_idx == 0:
                    raise CChanException(f"最高级别{lv}没有传入数据", ErrCode.NO_DATA)
                continue

            # 为每个K线单元设置级别类型
            for klu in inp[lv]:
                klu.kl_type = lv

            # 确保输入数据是列表类型
            assert isinstance(inp[lv], list)

            # 将输入数据转换为迭代器并添加到相应级别
            self.add_lv_iter(lv, iter(inp[lv]))

        # 从最高级别开始，逐级加载数据
        for _ in self.load_iterator(lv_idx=0, parent_klu=None, step=False):
            ...

        # 如果不是回放模式，在全部数据加载完成后计算中枢和线段
        if not self.conf.trigger_step:
            for lv in self.lv_list:
                self.kl_datas[lv].cal_seg_and_zs()

    def init_lv_klu_iter(self, stockapi_cls):
        # 初始化各个级别的K线单元迭代器
        # 目是跳过一些获取数据失败的级别
        lv_klu_iter = []
        valid_lv_list = []
        for lv in self.lv_list:
            try:
                # 尝试获取该级别的数据迭代器
                lv_klu_iter.append(self.get_load_stock_iter(stockapi_cls, lv))
                valid_lv_list.append(lv)
            except CChanException as e:
                # 如果是数据未找到错误，且配置允许自动跳过非法子级别
                if e.errcode == ErrCode.SRC_DATA_NOT_FOUND and self.conf.auto_skip_illegal_sub_lv:
                    if self.conf.print_warning:
                        print(f"[WARNING-{self.code}]{lv}级别获取数据失败，跳过")
                    del self.kl_datas[lv]
                    continue
                raise e
        self.lv_list = valid_lv_list
        return lv_klu_iter

    def GetStockAPI(self):
        # 根据数据源类型返回相应的API类
        _dict = {}
        # 初始化各种数据源的API类
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
        # 处理自定义数据源
        assert isinstance(self.data_src, str)
        if self.data_src.find("custom:") < 0:
            raise CChanException("load src type error", ErrCode.SRC_DATA_TYPE_ERR)
        package_info = self.data_src.split(":")[1]
        package_name, cls_name = package_info.split(".")
        exec(f"from DataAPI.{package_name} import {cls_name}")
        return eval(cls_name)

    def load(self, step=False):
        # 加载数据的主方法
        stockapi_cls = self.GetStockAPI()
        try:
            stockapi_cls.do_init()
            # 初始化各级别的迭代器
            for lv_idx, klu_iter in enumerate(self.init_lv_klu_iter(stockapi_cls)):
                self.add_lv_iter(lv_idx, klu_iter)
            self.klu_cache = [None for _ in self.lv_list]
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]

            # 开始加载数据
            yield from self.load_iterator(lv_idx=0, parent_klu=None, step=step)
            if not step:  # 非回放模式下，计算所有级别的中枢和线段
                for lv in self.lv_list:
                    self.kl_datas[lv].cal_seg_and_zs()
        except Exception:
            raise
        finally:
            stockapi_cls.do_close()
        if len(self[0]) == 0:
            raise CChanException("最高级别没有获得任何数据", ErrCode.NO_DATA)

    def set_klu_parent_relation(self, parent_klu, kline_unit, cur_lv, lv_idx):
        if self.conf.kl_data_check and kltype_lte_day(cur_lv) and kltype_lte_day(self.lv_list[lv_idx-1]):
            self.check_kl_consitent(parent_klu, kline_unit)
        parent_klu.add_children(kline_unit)
        kline_unit.set_parent(parent_klu)

    def add_new_kl(self, cur_lv: KL_TYPE, kline_unit):
        try:
            self.kl_datas[cur_lv].add_single_klu(kline_unit)
        except Exception:
            if self.conf.print_err_time:
                print(f"[ERROR-{self.code}]在计算{kline_unit.time}K线时发生错误!")
            raise

    def try_set_klu_idx(self, lv_idx: int, kline_unit: CKLine_Unit):
        if kline_unit.idx >= 0:
            return
        if len(self[lv_idx]) == 0:
            kline_unit.set_idx(0)
        else:
            kline_unit.set_idx(self[lv_idx][-1][-1].idx + 1)

    def load_iterator(self, lv_idx, parent_klu, step):
        # 递归加载各级别的K线数据
        # lv_idx: 当前处理的级别索引
        # parent_klu: 父级别的K线单元
        # step: 是否为步进模式

        # 获取当前级别
        cur_lv = self.lv_list[lv_idx]
        
        # 获取当前级别的最后一个K线单元，如果存在的话
        pre_klu = self[lv_idx][-1][-1] if len(self[lv_idx]) > 0 and len(self[lv_idx][-1]) > 0 else None

        while True:
            # 获取下一个K线单元
            if self.klu_cache[lv_idx]:
                # 如果缓存中有数据，直接使用缓存的K线单元
                kline_unit = self.klu_cache[lv_idx]
                assert kline_unit is not None
                self.klu_cache[lv_idx] = None
            else:
                try:
                    # 尝试获取下一个K线单元
                    kline_unit = self.get_next_lv_klu(lv_idx)
                    self.try_set_klu_idx(lv_idx, kline_unit)
                    
                    # 检查K线时间的单调性
                    if not kline_unit.time > self.klu_last_t[lv_idx]:
                        raise CChanException(f"kline time err, cur={kline_unit.time}, last={self.klu_last_t[lv_idx]}", ErrCode.KL_NOT_MONOTONOUS)
                    self.klu_last_t[lv_idx] = kline_unit.time
                except StopIteration:
                    # 如果没有更多的K线单元，结束循环
                    break

            # 处理父级别K线
            if parent_klu and kline_unit.time > parent_klu.time:
                # 如果当前K线单元的时间超过了父级别K线的时间，将其缓存并结束循环
                self.klu_cache[lv_idx] = kline_unit
                break
            
            # 设置K线关系并添加到数据结构中
            kline_unit.set_pre_klu(pre_klu)
            pre_klu = kline_unit
            self.add_new_kl(cur_lv, kline_unit)
            if parent_klu:
                self.set_klu_parent_relation(parent_klu, kline_unit, cur_lv, lv_idx)
            
            # 递归处理下一级别
            if lv_idx != len(self.lv_list)-1:
                # 如果不是最低级别，递归处理下一级别
                for _ in self.load_iterator(lv_idx+1, kline_unit, step):
                    ...  # 这里可能会进行一些处理，但在当前代码中是空的
                self.check_kl_align(kline_unit, lv_idx)
            
            # 如果是最高级别且为步进模式，则yield当前状态
            if lv_idx == 0 and step:
                yield self

    def check_kl_consitent(self, parent_klu, sub_klu):
        if parent_klu.time.year != sub_klu.time.year or \
           parent_klu.time.month != sub_klu.time.month or \
           parent_klu.time.day != sub_klu.time.day:
            self.kl_inconsistent_detail[str(parent_klu.time)].append(sub_klu.time)
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}]父级别时间是{parent_klu.time}，次级别时间却是{sub_klu.time}")
            if len(self.kl_inconsistent_detail) >= self.conf.max_kl_inconsistent_cnt:
                raise CChanException(f"父&子级别K线时间不一致条数超过{self.conf.max_kl_inconsistent_cnt}！！", ErrCode.KL_TIME_INCONSISTENT)

    def check_kl_align(self, kline_unit, lv_idx):
        if self.conf.kl_data_check and len(kline_unit.sub_kl_list) == 0:
            self.kl_misalign_cnt += 1
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}]当前{kline_unit.time}没在次级别{self.lv_list[lv_idx+1]}找到K线！！")
            if self.kl_misalign_cnt >= self.conf.max_kl_misalgin_cnt:
                raise CChanException(f"在次级别找不到K线条数超过{self.conf.max_kl_misalgin_cnt}！！", ErrCode.KL_DATA_NOT_ALIGN)

    def __getitem__(self, n) -> CKLine_List:
        if isinstance(n, KL_TYPE):
            return self.kl_datas[n]
        elif isinstance(n, int):
            return self.kl_datas[self.lv_list[n]]
        else:
            raise CChanException("unspoourt query type", ErrCode.COMMON_ERROR)

    def get_bsp(self, idx=None) -> List[CBS_Point]:
        if idx is not None:
            return sorted(self[idx].bs_point_lst.lst, key=lambda x: x.klu.time)
        assert len(self.lv_list) == 1
        return sorted(self[0].bs_point_lst.lst, key=lambda x: x.klu.time)

