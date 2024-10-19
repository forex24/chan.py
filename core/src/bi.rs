use crate::{
    bsp::CBS_Point,
    cenum::{BI_DIR, BI_TYPE, FX_TYPE},
    klc::CKLine,
    seg::CSeg,
};
use std::collections::HashMap;
/*
#[derive(Debug, Clone)]
pub struct CKLine {
    pub idx: i64,
    pub high: f64,
    pub low: f64,
    pub fx: FX_TYPE,
    pub next: Option<Box<CKLine>>,
    pub pre: Option<Box<CKLine>>,
    pub lst: Vec<CKLineUnit>,
    pub macd: CMACD,
    pub trade_info: CTradeInfo,
}

impl CKLine {
    pub pub fn get_peak_klu(&self, is_high: bool) -> CKLineUnit {
        if is_high {
            self.lst
                .iter()
                .max_by(|a, b| a.high.partial_cmp(&b.high).unwrap())
                .unwrap()
                .clone()
        } else {
            self.lst
                .iter()
                .min_by(|a, b| a.low.partial_cmp(&b.low).unwrap())
                .unwrap()
                .clone()
        }
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
*/
#[derive(Debug, Clone)]
pub struct CBi {
    pub begin_klc: CKLine,
    pub end_klc: CKLine,
    pub dir: Option<BI_DIR>,
    pub idx: i64,
    pub type_: BI_TYPE,
    pub is_sure: bool,
    pub sure_end: Vec<CKLine>,
    pub seg_idx: Option<i64>,
    pub parent_seg: Option<Box<CSeg<CBi>>>,
    pub bsp: Option<Box<CBS_Point<CBi>>>,
    pub next: Option<Box<CBi>>,
    pub pre: Option<Box<CBi>>,
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
        };
        bi.set(begin_klc, end_klc);
        bi
    }

    pub fn clean_cache(&mut self) {}

    pub fn begin_klc(&self) -> &CKLine {
        &self.begin_klc
    }

    pub fn end_klc(&self) -> &CKLine {
        &self.end_klc
    }

    pub fn dir(&self) -> &Option<BI_DIR> {
        &self.dir
    }

    pub fn idx(&self) -> i64 {
        self.idx
    }

    pub fn type_(&self) -> &BI_TYPE {
        &self.type_
    }

    pub fn is_sure(&self) -> bool {
        self.is_sure
    }

    pub fn sure_end(&self) -> &Vec<CKLine> {
        &self.sure_end
    }

    pub fn klc_lst(&self) -> Vec<CKLine> {
        let mut klc = self.begin_klc.clone();
        let mut lst = Vec::new();
        loop {
            lst.push(klc.clone());
            if let Some(next) = klc.next {
                klc = *next;
                if klc.idx > self.end_klc.idx {
                    break;
                }
            } else {
                break;
            }
        }
        lst
    }

    pub fn klc_lst_re(&self) -> Vec<CKLine> {
        let mut klc = self.end_klc.clone();
        let mut lst = Vec::new();
        loop {
            lst.push(klc.clone());
            if let Some(pre) = klc.pre {
                klc = *pre;
                if klc.idx < self.begin_klc.idx {
                    break;
                }
            } else {
                break;
            }
        }
        lst
    }

    pub fn seg_idx(&self) -> &Option<i64> {
        &self.seg_idx
    }

    pub fn set_seg_idx(&mut self, idx: i64) {
        self.seg_idx = Some(idx);
    }

    pub fn __str__(&self) -> String {
        format!(
            "{:?}|{} ~ {}",
            self.dir, self.begin_klc.idx, self.end_klc.idx
        )
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

    pub fn set(&mut self, begin_klc: CKLine, end_klc: CKLine) {
        self.begin_klc = begin_klc.clone();
        self.end_klc = end_klc.clone();
        self.dir = match begin_klc.fx {
            FX_TYPE::BOTTOM => Some(BI_DIR::UP),
            FX_TYPE::TOP => Some(BI_DIR::DOWN),
        };
        self.check();
        self.clean_cache();
    }

    pub fn get_begin_val(&self) -> f64 {
        if self.is_up() {
            self.begin_klc.low
        } else {
            self.begin_klc.high
        }
    }

    pub fn get_end_val(&self) -> f64 {
        if self.is_up() {
            self.end_klc.high
        } else {
            self.end_klc.low
        }
    }

    pub fn get_begin_klu(&self) -> CKLineUnit {
        if self.is_up() {
            self.begin_klc.get_peak_klu(false)
        } else {
            self.begin_klc.get_peak_klu(true)
        }
    }

    pub fn get_end_klu(&self) -> CKLineUnit {
        if self.is_up() {
            self.end_klc.get_peak_klu(true)
        } else {
            self.end_klc.get_peak_klu(false)
        }
    }

    pub fn amp(&self) -> f64 {
        (self.get_end_val() - self.get_begin_val()).abs()
    }

    pub fn get_klu_cnt(&self) -> i64 {
        self.get_end_klu().idx - self.get_begin_klu().idx + 1
    }

    pub fn get_klc_cnt(&self) -> i64 {
        assert_eq!(self.end_klc.idx, self.get_end_klu().idx);
        assert_eq!(self.begin_klc.idx, self.get_begin_klu().idx);
        self.end_klc.idx - self.begin_klc.idx + 1
    }

    pub fn _high(&self) -> f64 {
        if self.is_up() {
            self.end_klc.high
        } else {
            self.begin_klc.high
        }
    }

    pub fn _low(&self) -> f64 {
        if self.is_up() {
            self.begin_klc.low
        } else {
            self.end_klc.low
        }
    }

    pub fn _mid(&self) -> f64 {
        (self._high() + self._low()) / 2.0
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

    pub fn cal_macd_metric(&self, macd_algo: MACD_ALGO, is_reverse: bool) -> f64 {
        match macd_algo {
            MACD_ALGO::AREA => self.cal_macd_half(is_reverse),
            MACD_ALGO::PEAK => self.cal_macd_peak(),
            MACD_ALGO::FULL_AREA => self.cal_macd_area(),
            MACD_ALGO::DIFF => self.cal_macd_diff(),
            MACD_ALGO::SLOPE => self.cal_macd_slope(),
            MACD_ALGO::AMP => self.cal_macd_amp(),
            MACD_ALGO::AMOUNT => self.cal_macd_trade_metric(DATA_FIELD::FIELD_TURNOVER, false),
            MACD_ALGO::VOLUMN => self.cal_macd_trade_metric(DATA_FIELD::FIELD_VOLUME, false),
            MACD_ALGO::VOLUMN_AVG => self.cal_macd_trade_metric(DATA_FIELD::FIELD_VOLUME, true),
            MACD_ALGO::AMOUNT_AVG => self.cal_macd_trade_metric(DATA_FIELD::FIELD_TURNOVER, true),
            MACD_ALGO::TURNRATE_AVG => self.cal_macd_trade_metric(DATA_FIELD::FIELD_TURNRATE, true),
            MACD_ALGO::RSI => self.cal_rsi(),
            _ => panic!(
                "unsupport macd_algo={:?}, should be one of area/full_area/peak/diff/slope/amp",
                macd_algo
            ),
        }
    }

    pub fn cal_rsi(&self) -> f64 {
        let mut rsi_lst = Vec::new();
        for klc in self.klc_lst() {
            for klu in klc.lst {
                rsi_lst.push(klu.rsi);
            }
        }
        if self.is_down() {
            10000.0 / (rsi_lst.iter().cloned().fold(f64::INFINITY, f64::min) + 1e-7)
        } else {
            rsi_lst.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    }

    pub fn cal_macd_area(&self) -> f64 {
        let mut s = 1e-7;
        for klc in self.klc_lst() {
            for klu in klc.lst {
                s += klu.macd.macd.abs();
            }
        }
        s
    }

    pub fn cal_macd_peak(&self) -> f64 {
        let mut peak = 1e-7;
        for klc in self.klc_lst() {
            for klu in klc.lst {
                if klu.macd.macd.abs() > peak {
                    if self.is_down() && klu.macd.macd < 0.0 {
                        peak = klu.macd.macd.abs();
                    } else if self.is_up() && klu.macd.macd > 0.0 {
                        peak = klu.macd.macd.abs();
                    }
                }
            }
        }
        peak
    }

    pub fn cal_macd_half(&self, is_reverse: bool) -> f64 {
        if is_reverse {
            self.cal_macd_half_reverse()
        } else {
            self.cal_macd_half_obverse()
        }
    }

    pub fn cal_macd_half_obverse(&self) -> f64 {
        let mut s = 1e-7;
        let begin_klu = self.get_begin_klu();
        let peak_macd = begin_klu.macd.macd;
        for klc in self.klc_lst() {
            for klu in klc.lst {
                if klu.idx < begin_klu.idx {
                    continue;
                }
                if klu.macd.macd * peak_macd > 0.0 {
                    s += klu.macd.macd.abs();
                } else {
                    break;
                }
            }
        }
        s
    }

    pub fn cal_macd_half_reverse(&self) -> f64 {
        let mut s = 1e-7;
        let begin_klu = self.get_end_klu();
        let peak_macd = begin_klu.macd.macd;
        for klc in self.klc_lst_re() {
            for klu in klc.lst.iter().rev() {
                if klu.idx > begin_klu.idx {
                    continue;
                }
                if klu.macd.macd * peak_macd > 0.0 {
                    s += klu.macd.macd.abs();
                } else {
                    break;
                }
            }
        }
        s
    }

    pub fn cal_macd_diff(&self) -> f64 {
        let mut max = f64::NEG_INFINITY;
        let mut min = f64::INFINITY;
        for klc in self.klc_lst() {
            for klu in klc.lst {
                let macd = klu.macd.macd;
                if macd > max {
                    max = macd;
                }
                if macd < min {
                    min = macd;
                }
            }
        }
        max - min
    }

    pub fn cal_macd_slope(&self) -> f64 {
        let begin_klu = self.get_begin_klu();
        let end_klu = self.get_end_klu();
        if self.is_up() {
            (end_klu.high - begin_klu.low) / end_klu.high / (end_klu.idx - begin_klu.idx + 1) as f64
        } else {
            (begin_klu.high - end_klu.low)
                / begin_klu.high
                / (end_klu.idx - begin_klu.idx + 1) as f64
        }
    }

    pub fn cal_macd_amp(&self) -> f64 {
        let begin_klu = self.get_begin_klu();
        let end_klu = self.get_end_klu();
        if self.is_down() {
            (begin_klu.high - end_klu.low) / begin_klu.high
        } else {
            (end_klu.high - begin_klu.low) / begin_klu.low
        }
    }

    pub fn cal_macd_trade_metric(&self, metric: DATA_FIELD, cal_avg: bool) -> f64 {
        let mut s = 0.0;
        for klc in self.klc_lst() {
            for klu in klc.lst {
                let metric_res = match metric {
                    DATA_FIELD::FIELD_TURNOVER => klu
                        .trade_info
                        .metric
                        .get("turnover")
                        .cloned()
                        .unwrap_or(0.0),
                    DATA_FIELD::FIELD_VOLUME => {
                        klu.trade_info.metric.get("volume").cloned().unwrap_or(0.0)
                    }
                    DATA_FIELD::FIELD_TURNRATE => klu
                        .trade_info
                        .metric
                        .get("turnrate")
                        .cloned()
                        .unwrap_or(0.0),
                };
                s += metric_res;
            }
        }
        if cal_avg {
            s / self.get_klu_cnt() as f64
        } else {
            s
        }
    }
}
