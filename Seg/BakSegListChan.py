from Bi.BiList import CBiList
from Common.CEnum import BI_DIR, SEG_TYPE
from enum import auto, Enum
from typing import Optional, Tuple
import logging

from .EigenFX import CEigenFX
from .SegConfig import CSegConfig
from .SegListComm import CSegListComm


class SegState(Enum):
    INIT = auto()           # 初始状态
    COLLECTING = auto()     # 收集特征序列
    VALIDATING = auto()     # 验证特征序列
    CREATING_SEG = auto()   # 创建线段
    DONE = auto()           # 完成


class CSegListChan(CSegListComm):
    def __init__(self, seg_config=CSegConfig(), lv=SEG_TYPE.BI):
        super(CSegListChan, self).__init__(seg_config=seg_config, lv=lv)

    def do_init(self):
        # 删除末尾不确定的线段
        while len(self) and not self.lst[-1].is_sure:
            _seg = self[-1]
            for bi in _seg.bi_list:
                bi.parent_seg = None
            if _seg.pre:
                _seg.pre.next = None
            self.lst.pop()
        if len(self):
            assert self.lst[-1].eigen_fx and self.lst[-1].eigen_fx.ele[-1]
            if not self.lst[-1].eigen_fx.ele[-1].lst[-1].is_sure:
                # 如果确定线段的分形的第三元素包含不确定笔，也需要重新算，不然线段分形元素的高低点可能不对
                self.lst.pop()

    def update(self, bi_lst: CBiList):
        """更新线段列表"""
        logger = logging.getLogger(__name__)
        try:
            # 初始化处理
            self.do_init()
            
            # 计算确定的线段
            if len(self) == 0:
                logger.debug("Starting from beginning")
                self.cal_seg_sure(bi_lst, begin_idx=0)
            else:
                logger.debug(f"Starting from index {self[-1].end_bi.idx+1}")
                self.cal_seg_sure(bi_lst, begin_idx=self[-1].end_bi.idx+1)
            
            # 处理剩余的笔
            self.collect_left_seg(bi_lst)
            
        except Exception as e:
            logger.error("Error in update:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Current list length: {len(self)}")
            logger.error(f"Bi list length: {len(bi_lst)}")
            if len(self) > 0:
                logger.error(f"Last segment: {self[-1]}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    def cal_seg_sure(self, bi_lst: CBiList, begin_idx: int):
        """使用状态机处理笔的特征序列"""
        logger = logging.getLogger(__name__)
        state = SegState.INIT
        cur_idx = begin_idx
        up_eigen = CEigenFX(BI_DIR.UP, lv=self.lv)
        down_eigen = CEigenFX(BI_DIR.DOWN, lv=self.lv)
        last_seg_dir = None if len(self) == 0 else self[-1].dir
        active_eigen = None
        
        try:
            while state != SegState.DONE and cur_idx < len(bi_lst):
                bi = bi_lst[cur_idx]
                
                if state == SegState.INIT:
                    # 初始化特征序列
                    logger.debug(f"INIT: Processing bi {cur_idx} with direction {bi.dir}")
                    if bi.is_down() and last_seg_dir != BI_DIR.UP:
                        logger.debug(f"Selecting DOWN eigen for bi {cur_idx}")
                        active_eigen = down_eigen
                    elif bi.is_up() and last_seg_dir != BI_DIR.DOWN:
                        logger.debug(f"Selecting UP eigen for bi {cur_idx}")
                        active_eigen = up_eigen
                    else:
                        logger.debug(f"Skipping bi {cur_idx} due to direction conflict")
                        cur_idx += 1
                        continue
                    state = SegState.COLLECTING
                    
                elif state == SegState.COLLECTING:
                    try:
                        logger.debug(f"COLLECTING: Processing bi {cur_idx}")
                        # 检查是否已经有3个元素
                        if all(e is not None for e in active_eigen.ele):
                            logger.debug("Three elements found, moving to VALIDATING")
                            state = SegState.VALIDATING
                            continue
                            
                        # 检查方向是否相反
                        if bi.dir == active_eigen.dir:
                            logger.debug(f"Direction mismatch, skipping bi {cur_idx}")
                            cur_idx += 1
                            continue
                            
                        # 尝试添加笔
                        if active_eigen.add(bi):
                            logger.debug(f"Successfully added bi {cur_idx}")
                            if all(e is not None for e in active_eigen.ele):
                                state = SegState.VALIDATING
                            else:
                                cur_idx += 1
                        else:
                            logger.debug(f"Failed to add bi {cur_idx}")
                            cur_idx += 1
                            
                    except Exception as e:
                        logger.error(f"Error in COLLECTING state: {str(e)}")
                        logger.error(f"Active eigen elements: {[str(e) for e in active_eigen.ele]}")
                        raise
                        
                elif state == SegState.VALIDATING:
                    # 验证特征序列
                    logger.debug(f"VALIDATING: Testing eigen sequence at bi {cur_idx}")
                    test_result = active_eigen.can_be_end(bi_lst)
                    if test_result is True:
                        logger.debug("Validation successful, creating segment")
                        state = SegState.CREATING_SEG
                    elif test_result is False:
                        logger.debug("Validation failed, rolling back")
                        if active_eigen.lst[1] is not None:
                            cur_idx = active_eigen.lst[1].idx
                            active_eigen.clear()
                        state = SegState.INIT
                    else:  # test_result is None
                        logger.debug("Validation incomplete, continuing collection")
                        cur_idx += 1
                        state = SegState.COLLECTING
                        
                elif state == SegState.CREATING_SEG:
                    # 创建新线段
                    logger.debug(f"CREATING_SEG: Creating segment at bi {cur_idx}")
                    end_bi_idx = active_eigen.GetPeakBiIdx()
                    is_sure = active_eigen.all_bi_is_sure()
                    
                    # 检查方向一致性
                    if len(self) > 0 and self[-1].dir == active_eigen.dir:
                        logger.debug(f"Direction conflict with previous segment, resetting")
                        active_eigen.clear()
                        state = SegState.INIT
                        continue
                        
                    if self.add_new_seg(bi_lst, end_bi_idx, is_sure=is_sure):
                        logger.debug(f"Successfully created segment ending at bi {end_bi_idx}")
                        self.lst[-1].eigen_fx = active_eigen
                        cur_idx = end_bi_idx + 1
                        last_seg_dir = self[-1].dir
                    else:
                        logger.debug(f"Failed to create segment, moving to next bi")
                        cur_idx = end_bi_idx + 1
                    
                    # 重置状态
                    active_eigen.clear()
                    state = SegState.INIT
                    
        except Exception as e:
            logger.error("Error in cal_seg_sure:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Current idx: {cur_idx}")
            logger.error(f"Current state: {state}")
            if active_eigen:
                logger.error(f"Active eigen elements: {[str(e) for e in active_eigen.ele]}")
            raise
