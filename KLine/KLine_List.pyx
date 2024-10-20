# cython: language_level=3
import cython
from typing import List, Union, overload
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from Bi.Bi cimport CBi
from Bi.BiList cimport CBiList
from BuySellPoint.BSPointList cimport CBSPointList
from ChanConfig cimport CChanConfig
from Common.CEnum cimport KLINE_DIR, SEG_TYPE
from Common.ChanException cimport CChanException, ErrCode
from Seg.Seg cimport CSeg
from Seg.SegConfig cimport CSegConfig
from Seg.SegListComm cimport CSegListComm
from ZS.ZSList cimport CZSList

from .KLine cimport CKLine
from .KLine_Unit cimport CKLine_Unit

cdef CSegListComm get_seglist_instance(CSegConfig seg_config, SEG_TYPE lv):
    # 根据配置创建线段列表实例
    if seg_config.seg_algo == "chan":
        from Seg.SegListChan import CSegListChan
        return CSegListChan(seg_config, lv)
    elif seg_config.seg_algo == "1+1":
        print(f'Please avoid using seg_algo={seg_config.seg_algo} as it is deprecated and no longer maintained.')
        from Seg.SegListDYH import CSegListDYH
        return CSegListDYH(seg_config, lv)
    elif seg_config.seg_algo == "break":
        print(f'Please avoid using seg_algo={seg_config.seg_algo} as it is deprecated and no longer maintained.')
        from Seg.SegListDef import CSegListDef
        return CSegListDef(seg_config, lv)
    else:
        raise CChanException(f"unsupport seg algoright:{seg_config.seg_algo}", ErrCode.PARA_ERROR)

cdef class CKLine_List:
    cdef:
        public object kl_type
        public CChanConfig config
        public list lst
        public CBiList bi_list
        public CSegListComm seg_list
        public CSegListComm segseg_list
        public CZSList zs_list
        public CZSList segzs_list
        public CBSPointList bs_point_lst
        public CBSPointList seg_bs_point_lst
        public list metric_model_lst
        public bint step_calculation

    def __cinit__(self, kl_type, CChanConfig conf):
        self.kl_type = kl_type
        self.config = conf
        self.lst = []  # K线列表，可递归，元素为KLine类型
        self.bi_list = CBiList(bi_conf=conf.bi_conf)  # 笔列表
        self.seg_list = get_seglist_instance(seg_config=conf.seg_conf, lv=SEG_TYPE.BI)  # 线段列表
        self.segseg_list = get_seglist_instance(seg_config=conf.seg_conf, lv=SEG_TYPE.SEG)  # 线段的线段列表
        
        self.zs_list = CZSList(zs_config=conf.zs_conf)  # 中枢列表
        self.segzs_list = CZSList(zs_config=conf.zs_conf)  # 线段中枢列表
        
        self.bs_point_lst = CBSPointList[CBi, CBiList](bs_point_config=conf.bs_point_conf)  # 买卖点列表
        self.seg_bs_point_lst = CBSPointList[CSeg, CSegListComm](bs_point_config=conf.seg_bs_point_conf)  # 线段买卖点列表
        
        self.metric_model_lst = conf.GetMetricModel()  # 指标模型列表
        
        self.step_calculation = self.need_cal_step_by_step()  # 是否需要逐步计算

    def __deepcopy__(self, memo):
        # 深拷贝方法，用于创建对象的完整副本
        cdef:
            CKLine_List new_obj = CKLine_List(self.kl_type, self.config)
            CKLine klc, new_klc
            CKLine_Unit klu, new_klu
            int idx
        
        memo[id(self)] = new_obj
        for klc in self.lst:
            klus_new = []
            for klu in klc.lst:
                new_klu = klu.__class__.__new__(klu.__class__)
                new_klu.__dict__.update(klu.__dict__)
                memo[id(klu)] = new_klu
                if klu.pre is not None:
                    new_klu.set_pre_klu(memo[id(klu.pre)])
                klus_new.append(new_klu)

            new_klc = CKLine(klus_new[0], idx=klc.idx, _dir=klc.dir)
            new_klc.set_fx(klc.fx)
            new_klc.kl_type = klc.kl_type
            for idx, klu in enumerate(klus_new):
                klu.set_klc(new_klc)
                if idx != 0:
                    new_klc.add(klu)
            memo[id(klc)] = new_klc
            if new_obj.lst:
                new_obj.lst[-1].set_next(new_klc)
                new_klc.set_pre(new_obj.lst[-1])
            new_obj.lst.append(new_klc)
        new_obj.bi_list = self.bi_list.__class__.__new__(self.bi_list.__class__)
        new_obj.bi_list.__dict__.update(self.bi_list.__dict__)
        new_obj.seg_list = self.seg_list.__class__.__new__(self.seg_list.__class__)
        new_obj.seg_list.__dict__.update(self.seg_list.__dict__)
        new_obj.segseg_list = self.segseg_list.__class__.__new__(self.segseg_list.__class__)
        new_obj.segseg_list.__dict__.update(self.segseg_list.__dict__)
        new_obj.zs_list = self.zs_list.__class__.__new__(self.zs_list.__class__)
        new_obj.zs_list.__dict__.update(self.zs_list.__dict__)
        new_obj.segzs_list = self.segzs_list.__class__.__new__(self.segzs_list.__class__)
        new_obj.segzs_list.__dict__.update(self.segzs_list.__dict__)
        new_obj.bs_point_lst = self.bs_point_lst.__class__.__new__(self.bs_point_lst.__class__)
        new_obj.bs_point_lst.__dict__.update(self.bs_point_lst.__dict__)
        new_obj.metric_model_lst = self.metric_model_lst[:]
        new_obj.step_calculation = self.step_calculation
        new_obj.seg_bs_point_lst = self.seg_bs_point_lst.__class__.__new__(self.seg_bs_point_lst.__class__)
        new_obj.seg_bs_point_lst.__dict__.update(self.seg_bs_point_lst.__dict__)
        return new_obj

    @overload
    def __getitem__(self, index: int) -> CKLine: ...

    @overload
    def __getitem__(self, index: slice) -> List[CKLine]: ...

    def __getitem__(self, index: Union[slice, int]) -> Union[List[CKLine], CKLine]:
        return self.lst[index]

    def __len__(self):
        return len(self.lst)

    cpdef void cal_seg_and_zs(self):
        # 计算线段和中枢
        if not self.step_calculation:
            self.bi_list.try_add_virtual_bi(self.lst[-1])
        cal_seg(self.bi_list, self.seg_list)
        self.zs_list.cal_bi_zs(self.bi_list, self.seg_list)
        update_zs_in_seg(self.bi_list, self.seg_list, self.zs_list)  # 计算seg的zs_lst，以及中枢的bi_in, bi_out

        cal_seg(self.seg_list, self.segseg_list)
        self.segzs_list.cal_bi_zs(self.seg_list, self.segseg_list)
        update_zs_in_seg(self.seg_list, self.segseg_list, self.segzs_list)  # 计算segseg的zs_lst，以及中枢的bi_in, bi_out

        # 计算买卖点
        self.seg_bs_point_lst.cal(self.seg_list, self.segseg_list)  # 线段线段买卖点
        self.bs_point_lst.cal(self.bi_list, self.seg_list)  # 再算笔买卖点

    cpdef bint need_cal_step_by_step(self):
        return self.config.trigger_step

    cpdef void add_single_klu(self, CKLine_Unit klu):
        # 添加单个K线单元
        klu.set_metric(self.metric_model_lst)
        if len(self.lst) == 0:
            self.lst.append(CKLine(klu, idx=0))
        else:
            _dir = self.lst[-1].try_add(klu)
            if _dir != KLINE_DIR.COMBINE:  # 不需要合并K线
                self.lst.append(CKLine(klu, idx=len(self.lst), _dir=_dir))
                if len(self.lst) >= 3:
                    self.lst[-2].update_fx(self.lst[-3], self.lst[-1])
                if self.bi_list.update_bi(self.lst[-2], self.lst[-1], self.step_calculation) and self.step_calculation:
                    self.cal_seg_and_zs()
            elif self.step_calculation and self.bi_list.try_add_virtual_bi(self.lst[-1], need_del_end=True):
                self.cal_seg_and_zs()

    def klu_iter(self, int klc_begin_idx=0):
        # K线单元迭代器
        cdef:
            CKLine klc
            CKLine_Unit klu
        for klc in self.lst[klc_begin_idx:]:
            for klu in klc.lst:
                yield klu

cdef void cal_seg(CBiList bi_list, CSegListComm seg_list):
    # 计算线段
    cdef:
        int sure_seg_cnt = 0
        CSeg begin_seg, seg, cur_seg
        CBi bi

    seg_list.update(bi_list)

    if len(seg_list) == 0:
        for bi in bi_list:
            bi.set_seg_idx(0)
        return

    begin_seg = seg_list[-1]
    for seg in seg_list[::-1]:
        if seg.is_sure:
            sure_seg_cnt += 1
        else:
            sure_seg_cnt = 0
        begin_seg = seg
        if sure_seg_cnt > 2:
            break

    cur_seg = seg_list[-1]
    for bi in bi_list[::-1]:
        if bi.seg_idx is not None and bi.idx < begin_seg.start_bi.idx:
            break
        if bi.idx > cur_seg.end_bi.idx:
            bi.set_seg_idx(cur_seg.idx+1)
            continue
        if bi.idx < cur_seg.start_bi.idx:
            assert cur_seg.pre
            cur_seg = cur_seg.pre
        bi.set_seg_idx(cur_seg.idx)

cdef void update_zs_in_seg(CBiList bi_list, CSegListComm seg_list, CZSList zs_list):
    # 更新线段中的中枢
    cdef:
        int sure_seg_cnt = 0
        CSeg seg
        CZS zs

    for seg in seg_list[::-1]:
        if seg.ele_inside_is_sure:
            break
        if seg.is_sure:
            sure_seg_cnt += 1
        seg.clear_zs_lst()
        for zs in zs_list[::-1]:
            if zs.end.idx < seg.start_bi.get_begin_klu().idx:
                break
            if zs.is_inside(seg):
                seg.add_zs(zs)
            assert zs.begin_bi.idx > 0
            zs.set_bi_in(bi_list[zs.begin_bi.idx-1])
            if zs.end_bi.idx+1 < len(bi_list):
                zs.set_bi_out(bi_list[zs.end_bi.idx+1])
            zs.set_bi_lst(list(bi_list[zs.begin_bi.idx:zs.end_bi.idx+1]))

        if sure_seg_cnt > 2:
            if not seg.ele_inside_is_sure:
                seg.ele_inside_is_sure = True
