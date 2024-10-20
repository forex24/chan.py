use std::collections::HashMap;

use crate::{
    bi::CBi,
    cenum::{BI_DIR, BI_TYPE, FX_TYPE, KLINE_DIR},
    klc::CKLine,
};

/*
#[derive(Debug, Clone)]
struct CChanException {
    message: String,
    err_code: i32,
}

impl CChanException {
    pub fn new(message: String, err_code: i32) -> Self {
        Self { message, err_code }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum FX_TYPE {
    UNKNOWN,
    BOTTOM,
    TOP,
}

#[derive(Debug, Clone, PartialEq)]
enum KLINE_DIR {
    UP,
    DOWN,
}

#[derive(Debug, Clone)]
struct CKLine {
    idx: i64,
    high: f64,
    low: f64,
    fx: FX_TYPE,
    next: Option<Box<CKLine>>,
    pre: Option<Box<CKLine>>,
    lst: Vec<CKLineUnit>,
    macd: CMACD,
    trade_info: CTradeInfo,
}

impl CKLine {
    pub fn get_next(&self) -> Option<&CKLine> {
        self.next.as_ref().map(|boxed| boxed.as_ref())
    }

    pub fn check_fx_valid(&self, klc: &CKLine, bi_fx_check: bool, for_virtual: bool) -> bool {
        // Implement this method based on your logic
        true
    }

    pub fn has_gap_with_next(&self) -> bool {
        // Implement this method based on your logic
        false
    }
}

#[derive(Debug, Clone)]
struct CKLineUnit {
    idx: i64,
    high: f64,
    low: f64,
    macd: CMACD,
    rsi: f64,
}

#[derive(Debug, Clone)]
struct CMACD {
    macd: f64,
}

#[derive(Debug, Clone)]
struct CTradeInfo {
    metric: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct CBi {
    begin_klc: CKLine,
    end_klc: CKLine,
    dir: Option<BI_DIR>,
    idx: i64,
    type_: BI_TYPE,
    is_sure: bool,
    sure_end: Vec<CKLine>,
    seg_idx: Option<i64>,
    parent_seg: Option<CSeg<CBi>>,
    bsp: Option<CBS_Point<CBi>>,
    next: Option<Box<CBi>>,
    pre: Option<Box<CBi>>,
    _memoize_cache: HashMap<String, f64>,
}

impl CBi {
    pub fn new(begin_klc: CKLine, end_klc: CKLine, idx: i64, is_sure: bool) -> Self {
        let mut bi = Self {
            begin_klc: begin_klc.clone(),
            end_klc: end_klc.clone(),
            dir: None,
            idx,
            type_: BI_TYPE::STRICT,
            is_sure,
            sure_end: Vec::new(),
            seg_idx: None,
            parent_seg: None,
            bsp: None,
            next: None,
            pre: None,
            _memoize_cache: HashMap::new(),
        };
        bi.set(begin_klc, end_klc);
        bi
    }

    pub fn set(&mut self, begin_klc: CKLine, end_klc: CKLine) {
        self.begin_klc = begin_klc.clone();
        self.end_klc = end_klc.clone();
        self.dir = match begin_klc.fx {
            FX_TYPE::BOTTOM => Some(BI_DIR::UP),
            FX_TYPE::TOP => Some(BI_DIR::DOWN),
            _ => None,
        };
        self.check();
        self.clean_cache();
    }

    pub fn check(&self) {
        if self.is_down() {
            assert!(
                self.begin_klc.high > self.end_klc.low,
                "笔的方向和收尾位置不一致!"
            );
        } else {
            assert!(
                self.begin_klc.low < self.end_klc.high,
                "笔的方向和收尾位置不一致!"
            );
        }
    }

    pub fn clean_cache(&mut self) {
        self._memoize_cache.clear();
    }

    pub fn get_end_klu(&self) -> CKLineUnit {
        if self.is_up() {
            self.end_klc.get_peak_klu(true)
        } else {
            self.end_klc.get_peak_klu(false)
        }
    }

    pub fn is_down(&self) -> bool {
        self.dir == Some(BI_DIR::DOWN)
    }

    pub fn is_up(&self) -> bool {
        self.dir == Some(BI_DIR::UP)
    }

    pub fn update_virtual_end(&mut self, new_klc: CKLine) {
        self.append_sure_end(self.end_klc.clone());
        self.update_new_end(new_klc);
        self.is_sure = false;
    }

    pub fn restore_from_virtual_end(&mut self, sure_end: CKLine) {
        self.is_sure = true;
        self.update_new_end(sure_end);
        self.sure_end.clear();
    }

    pub fn append_sure_end(&mut self, klc: CKLine) {
        self.sure_end.push(klc);
    }

    pub fn update_new_end(&mut self, new_klc: CKLine) {
        self.end_klc = new_klc;
        self.check();
        self.clean_cache();
    }
}
*/

#[derive(Debug, Clone)]
pub struct CBiConfig {
    pub bi_allow_sub_peak: bool,
    pub is_strict: bool,
    pub gap_as_kl: bool,
    pub bi_algo: String,
    pub bi_fx_check: bool,
    pub bi_end_is_peak: bool,
}

impl CBiConfig {
    pub fn new() -> Self {
        Self {
            bi_allow_sub_peak: false,
            is_strict: false,
            gap_as_kl: false,
            bi_algo: "fx".to_string(),
            bi_fx_check: false,
            bi_end_is_peak: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CBiList {
    pub bi_list: Vec<CBi>,
    pub last_end: Option<CKLine>,
    pub config: CBiConfig,
    pub free_klc_lst: Vec<CKLine>,
}

impl CBiList {
    pub fn new(bi_conf: CBiConfig) -> Self {
        Self {
            bi_list: Vec::new(),
            last_end: None,
            config: bi_conf,
            free_klc_lst: Vec::new(),
        }
    }

    pub fn __str__(&self) -> String {
        self.bi_list
            .iter()
            .map(|bi| bi.__str__())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn __iter__(&self) -> std::slice::Iter<CBi> {
        self.bi_list.iter()
    }

    pub fn __getitem__(&self, index: usize) -> &CBi {
        &self.bi_list[index]
    }

    pub fn __len__(&self) -> usize {
        self.bi_list.len()
    }

    pub fn try_create_first_bi(&mut self, klc: CKLine) -> bool {
        for exist_free_klc in &self.free_klc_lst {
            if exist_free_klc.fx == klc.fx {
                continue;
            }
            if self.can_make_bi(klc.clone(), exist_free_klc.clone()) {
                self.add_new_bi(exist_free_klc.clone(), klc.clone());
                self.last_end = Some(klc.clone());
                return true;
            }
        }
        self.free_klc_lst.push(klc.clone());
        self.last_end = Some(klc.clone());
        return false;
    }

    pub fn update_bi(&mut self, klc: CKLine, last_klc: CKLine, cal_virtual: bool) -> bool {
        let flag1 = self.update_bi_sure(klc.clone());
        if cal_virtual {
            let flag2 = self.try_add_virtual_bi(last_klc.clone());
            return flag1 || flag2;
        } else {
            return flag1;
        }
    }

    pub fn can_update_peak(&self, klc: &CKLine) -> bool {
        if self.config.bi_allow_sub_peak || self.bi_list.len() < 2 {
            return false;
        }
        if self.bi_list.last().unwrap().is_down()
            && klc.high < self.bi_list.last().unwrap().begin_klc.high
        {
            return false;
        }
        if self.bi_list.last().unwrap().is_up()
            && klc.low > self.bi_list.last().unwrap().begin_klc.low
        {
            return false;
        }
        if !end_is_peak(
            self.bi_list[self.bi_list.len() - 2].begin_klc.clone(),
            klc.clone(),
        ) {
            return false;
        }
        if self.bi_list.last().unwrap().is_down()
            && self.bi_list.last().unwrap().end_klc.low
                < self.bi_list[self.bi_list.len() - 2].begin_klc.high
        {
            return false;
        }
        if self.bi_list.last().unwrap().is_up()
            && self.bi_list.last().unwrap().end_klc.high
                > self.bi_list[self.bi_list.len() - 2].begin_klc.low
        {
            return false;
        }
        return true;
    }

    pub fn update_peak(&mut self, klc: CKLine, for_virtual: bool) -> bool {
        if !self.can_update_peak(&klc) {
            return false;
        }
        let _tmp_last_bi = self.bi_list.pop().unwrap();
        if !self.try_update_end(klc.clone(), for_virtual) {
            self.bi_list.push(_tmp_last_bi);
            return false;
        } else {
            if for_virtual {
                self.bi_list
                    .last_mut()
                    .unwrap()
                    .append_sure_end(_tmp_last_bi.end_klc.clone());
            }
            return true;
        }
    }

    pub fn update_bi_sure(&mut self, klc: CKLine) -> bool {
        let _tmp_end = self.get_last_klu_of_last_bi();
        self.delete_virtual_bi();
        if klc.fx == FX_TYPE::UNKNOWN {
            return _tmp_end != self.get_last_klu_of_last_bi();
        }
        if self.last_end.is_none() || self.bi_list.is_empty() {
            return self.try_create_first_bi(klc.clone());
        }
        if klc.fx == self.last_end.as_ref().unwrap().fx {
            return self.try_update_end(klc.clone());
        } else if self.can_make_bi(klc.clone(), self.last_end.as_ref().unwrap().clone()) {
            self.add_new_bi(self.last_end.as_ref().unwrap().clone(), klc.clone());
            self.last_end = Some(klc.clone());
            return true;
        } else if self.update_peak(klc.clone(), false) {
            return true;
        }
        return _tmp_end != self.get_last_klu_of_last_bi();
    }

    pub fn delete_virtual_bi(&mut self) {
        if !self.bi_list.is_empty() && !self.bi_list.last().unwrap().is_sure {
            let sure_end_list: Vec<CKLine> = self.bi_list.last().unwrap().sure_end.clone();
            if !sure_end_list.is_empty() {
                self.bi_list
                    .last_mut()
                    .unwrap()
                    .restore_from_virtual_end(sure_end_list[0].clone());
                self.last_end = Some(self.bi_list.last().unwrap().end_klc.clone());
                for sure_end in sure_end_list.iter().skip(1) {
                    self.add_new_bi(
                        self.last_end.as_ref().unwrap().clone(),
                        sure_end.clone(),
                        true,
                    );
                    self.last_end = Some(self.bi_list.last().unwrap().end_klc.clone());
                }
            } else {
                self.bi_list.pop();
            }
        }
        self.last_end = if !self.bi_list.is_empty() {
            Some(self.bi_list.last().unwrap().end_klc.clone())
        } else {
            None
        };
        if !self.bi_list.is_empty() {
            self.bi_list.last_mut().unwrap().next = None;
        }
    }

    pub fn try_add_virtual_bi(&mut self, klc: CKLine, need_del_end: bool) -> bool {
        if need_del_end {
            self.delete_virtual_bi();
        }
        if self.bi_list.is_empty() {
            return false;
        }
        if klc.idx == self.bi_list.last().unwrap().end_klc.idx {
            return false;
        }
        if (self.bi_list.last().unwrap().is_up()
            && klc.high >= self.bi_list.last().unwrap().end_klc.high)
            || (self.bi_list.last().unwrap().is_down()
                && klc.low <= self.bi_list.last().unwrap().end_klc.low)
        {
            self.bi_list
                .last_mut()
                .unwrap()
                .update_virtual_end(klc.clone());
            return true;
        }
        let mut _tmp_klc = klc.clone();
        while _tmp_klc.is_some()
            && _tmp_klc.as_ref().unwrap().idx > self.bi_list.last().unwrap().end_klc.idx
        {
            if self.can_make_bi(
                _tmp_klc.as_ref().unwrap().clone(),
                self.bi_list.last().unwrap().end_klc.clone(),
                true,
            ) {
                self.add_new_bi(
                    self.last_end.as_ref().unwrap().clone(),
                    _tmp_klc.as_ref().unwrap().clone(),
                    false,
                );
                return true;
            } else if self.update_peak(_tmp_klc.as_ref().unwrap().clone(), true) {
                return true;
            }
            _tmp_klc = _tmp_klc.as_ref().unwrap().pre.clone();
        }
        return false;
    }

    pub fn add_new_bi(&mut self, pre_klc: CKLine, cur_klc: CKLine, is_sure: bool) {
        self.bi_list.push(CBi::new(
            pre_klc.clone(),
            cur_klc.clone(),
            self.bi_list.len() as i64,
            is_sure,
        ));
        if self.bi_list.len() >= 2 {
            self.bi_list[self.bi_list.len() - 2].next =
                Some(Box::new(self.bi_list.last().unwrap().clone()));
            self.bi_list.last_mut().unwrap().pre =
                Some(Box::new(self.bi_list[self.bi_list.len() - 2].clone()));
        }
    }

    pub fn satisfy_bi_span(&self, klc: &CKLine, last_end: &CKLine) -> bool {
        let bi_span = self.get_klc_span(klc, last_end);
        if self.config.is_strict {
            return bi_span >= 4;
        }
        let mut uint_kl_cnt = 0;
        let mut tmp_klc = last_end.next.as_ref().map(|boxed| boxed.as_ref());
        while tmp_klc.is_some() {
            uint_kl_cnt += tmp_klc.as_ref().unwrap().lst.len();
            if tmp_klc.as_ref().unwrap().next.is_none() {
                return false;
            }
            if tmp_klc.as_ref().unwrap().next.as_ref().unwrap().idx < klc.idx {
                tmp_klc = tmp_klc.unwrap().next.as_ref().map(|boxed| boxed.as_ref());
            } else {
                break;
            }
        }
        return bi_span >= 3 && uint_kl_cnt >= 3;
    }

    pub fn get_klc_span(&self, klc: &CKLine, last_end: &CKLine) -> i64 {
        let mut span = klc.idx - last_end.idx;
        if !self.config.gap_as_kl {
            return span;
        }
        if span >= 4 {
            return span;
        }
        let mut tmp_klc = last_end.clone();
        while tmp_klc.is_some() && tmp_klc.as_ref().unwrap().idx < klc.idx {
            if tmp_klc.as_ref().unwrap().has_gap_with_next() {
                span += 1;
            }
            tmp_klc = tmp_klc.unwrap().next.clone();
        }
        return span;
    }

    pub fn can_make_bi(&self, klc: CKLine, last_end: CKLine, for_virtual: bool) -> bool {
        let satisify_span = if self.config.bi_algo == "fx" {
            true
        } else {
            self.satisfy_bi_span(&klc, &last_end)
        };
        if !satisify_span {
            return false;
        }
        if !last_end.check_fx_valid(&klc, self.config.bi_fx_check, for_virtual) {
            return false;
        }
        if self.config.bi_end_is_peak && !end_is_peak(last_end, klc) {
            return false;
        }
        return true;
    }

    pub fn try_update_end(&mut self, klc: CKLine, for_virtual: bool) -> bool {
        pub fn check_top(klc: &CKLine, for_virtual: bool) -> bool {
            if for_virtual {
                klc.dir == KLINE_DIR::UP
            } else {
                klc.fx == FX_TYPE::TOP
            }
        }

        pub fn check_bottom(klc: &CKLine, for_virtual: bool) -> bool {
            if for_virtual {
                klc.dir == KLINE_DIR::DOWN
            } else {
                klc.fx == FX_TYPE::BOTTOM
            }
        }

        if self.bi_list.is_empty() {
            return false;
        }
        let last_bi = self.bi_list.last_mut().unwrap();
        if (last_bi.is_up() && check_top(&klc, for_virtual) && klc.high >= last_bi.end_klc.high)
            || (last_bi.is_down()
                && check_bottom(&klc, for_virtual)
                && klc.low <= last_bi.end_klc.low)
        {
            if for_virtual {
                last_bi.update_virtual_end(klc.clone());
            } else {
                last_bi.update_new_end(klc.clone());
            }
            self.last_end = Some(klc.clone());
            return true;
        } else {
            return false;
        }
    }

    pub fn get_last_klu_of_last_bi(&self) -> Option<i64> {
        if !self.bi_list.is_empty() {
            Some(self.bi_list.last().unwrap().get_end_klu().idx)
        } else {
            None
        }
    }
}

pub fn end_is_peak(last_end: CKLine, cur_end: CKLine) -> bool {
    if last_end.fx == FX_TYPE::BOTTOM {
        let cmp_thred = cur_end.high;
        let mut klc = last_end.get_next().unwrap();
        while klc.idx < cur_end.idx {
            if klc.high > cmp_thred {
                return false;
            }
            klc = klc.get_next().unwrap();
        }
    } else if last_end.fx == FX_TYPE::TOP {
        let cmp_thred = cur_end.low;
        let mut klc = last_end.get_next().unwrap();
        while klc.idx < cur_end.idx {
            if klc.low < cmp_thred {
                return false;
            }
            klc = klc.get_next().unwrap();
        }
    }
    return true;
}
