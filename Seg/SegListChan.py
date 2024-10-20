from Bi.BiList import CBiList
from Common.CEnum import BI_DIR, SEG_TYPE

from .EigenFX import CEigenFX
from .SegConfig import CSegConfig
from .SegListComm import CSegListComm


class CSegListChan(CSegListComm):
    def __init__(self, seg_config=CSegConfig(), lv=SEG_TYPE.BI):
        # 初始化方法，设置线段配置和级别
        super(CSegListChan, self).__init__(seg_config=seg_config, lv=lv)

    def do_init(self):
        # 初始化方法，删除末尾不确定的线段
        while len(self) and not self.lst[-1].is_sure:
            _seg = self[-1]
            for bi in _seg.bi_list:
                bi.parent_seg = None  # 解除笔与线段的关联
            if _seg.pre:
                _seg.pre.next = None  # 解除线段间的链接
            self.lst.pop()  # 移除不确定的线段
        
        # 如果最后一个确定线段的特征分型的最后一个元素包含不确定的笔，也需要重新计算
        if len(self):
            assert self.lst[-1].eigen_fx and self.lst[-1].eigen_fx.ele[-1]
            if not self.lst[-1].eigen_fx.ele[-1].lst[-1].is_sure:
                self.lst.pop()

    def update(self, bi_lst: CBiList):
        # 更新线段列表
        self.do_init()  # 初始化，删除不确定的线段
        if len(self) == 0:
            self.cal_seg_sure(bi_lst, begin_idx=0)  # 如果没有线段，从头开始计算
        else:
            self.cal_seg_sure(bi_lst, begin_idx=self[-1].end_bi.idx+1)  # 从最后一个确定线段的结束笔之后开始计算
        self.collect_left_seg(bi_lst)  # 收集剩余的可能形成线段的笔

    def cal_seg_sure(self, bi_lst: CBiList, begin_idx: int):
        # 计算确定的线段
        up_eigen = CEigenFX(BI_DIR.UP, lv=self.lv)  # 上升线段的特征分型
        down_eigen = CEigenFX(BI_DIR.DOWN, lv=self.lv)  # 下降线段的特征分型
        last_seg_dir = None if len(self) == 0 else self[-1].dir  # 最后一个线段的方向
        
        for bi in bi_lst[begin_idx:]:
            fx_eigen = None
            # 判断当前笔的方向，并尝试添加到相应的特征分型中
            if bi.is_down() and last_seg_dir != BI_DIR.UP:
                if up_eigen.add(bi):
                    fx_eigen = up_eigen
            elif bi.is_up() and last_seg_dir != BI_DIR.DOWN:
                if down_eigen.add(bi):
                    fx_eigen = down_eigen
            
            # 尝试确定第一段方向
            if len(self) == 0:
                if up_eigen.ele[1] is not None and bi.is_down():
                    last_seg_dir = BI_DIR.DOWN
                    down_eigen.clear()
                elif down_eigen.ele[1] is not None and bi.is_up():
                    up_eigen.clear()
                    last_seg_dir = BI_DIR.UP
                if up_eigen.ele[1] is None and last_seg_dir == BI_DIR.DOWN and bi.dir == BI_DIR.DOWN:
                    last_seg_dir = None
                elif down_eigen.ele[1] is None and last_seg_dir == BI_DIR.UP and bi.dir == BI_DIR.UP:
                    last_seg_dir = None

            # 如果形成了特征分型，进行处理
            if fx_eigen:
                self.treat_fx_eigen(fx_eigen, bi_lst)
                break

    def treat_fx_eigen(self, fx_eigen, bi_lst: CBiList):
        # 处理特征分型
        _test = fx_eigen.can_be_end(bi_lst)  # 检查是否可以作为线段的结束
        end_bi_idx = fx_eigen.GetPeakBiIdx()  # 获取峰值笔的索引
        if _test in [True, None]:  # None表示反向分型找到尾部也没找到
            is_true = _test is not None  # 如果是正常结束
            # 尝试添加新的线段
            if not self.add_new_seg(bi_lst, end_bi_idx, is_sure=is_true and fx_eigen.all_bi_is_sure()):
                self.cal_seg_sure(bi_lst, end_bi_idx+1)
                return
            self.lst[-1].eigen_fx = fx_eigen  # 设置线段的特征分型
            if is_true:
                self.cal_seg_sure(bi_lst, end_bi_idx + 1)  # 继续计算下一个线段
        else:
            self.cal_seg_sure(bi_lst, fx_eigen.lst[1].idx)  # 从特征分型的第二个元素开始重新计算
