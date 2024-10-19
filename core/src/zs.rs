use std::collections::HashMap;

use crate::klu::CKLine_Unit;

/*#[derive(Debug, Clone)]
struct CChanException {
    message: String,
    err_code: i32,
}

impl CChanException {
    pub fn new(message: String, err_code: i32) -> Self {
        Self { message, err_code }
    }
}

#[derive(Debug, Clone)]
struct CBi {
    begin_klc: CKLineUnit,
    end_klc: CKLineUnit,
    high: f64,
    low: f64,
    dir: i32,
    idx: i64,
    next: Option<Box<CBi>>,
    is_sure: bool,
}

impl CBi {
    pub fn _high(&self) -> f64 {
        self.high
    }

    pub fn _low(&self) -> f64 {
        self.low
    }

    pub fn is_down(&self) -> bool {
        self.dir == 1
    }

    pub fn is_up(&self) -> bool {
        self.dir == -1
    }

    pub fn get_end_val(&self) -> f64 {
        self.end_klc.high
    }

    pub fn get_begin_val(&self) -> f64 {
        self.begin_klc.high
    }

    pub fn get_end_klu(&self) -> CKLineUnit {
        self.end_klc.clone()
    }

    pub fn get_begin_klu(&self) -> CKLineUnit {
        self.begin_klc.clone()
    }
}

#[derive(Debug, Clone)]
struct CKLineUnit {
    time: i64,
    high: f64,
    low: f64,
}

#[derive(Debug, Clone)]
struct CSeg<T: Clone> {
    idx: i64,
    start_bi: T,
    end_bi: T,
    is_sure: bool,
    dir: i32,
    zs_lst: Vec<CZS<T>>,
    eigen_fx: Option<CEigenFX>,
    seg_idx: Option<i64>,
    parent_seg: Option<Box<CSeg<T>>>,
    pre: Option<Box<CSeg<T>>>,
    next: Option<Box<CSeg<T>>>,
    bsp: Option<CBS_Point>,
    bi_list: Vec<T>,
    reason: String,
    support_trend_line: Option<CTrendLine>,
    resistance_trend_line: Option<CTrendLine>,
    ele_inside_is_sure: bool,
}

impl<T: Clone + PartialEq> CSeg<T> {
    pub fn new(idx: i64, start_bi: T, end_bi: T, is_sure: bool, seg_dir: Option<i32>, reason: &str) -> Result<Self, CChanException> {
        assert!(start_bi.idx == 0 || start_bi.dir == end_bi.dir || !is_sure, format!("{} {} {} {}", start_bi.idx, end_bi.idx, start_bi.dir, end_bi.dir));
        let dir = seg_dir.unwrap_or(end_bi.dir);
        let mut seg = Self {
            idx,
            start_bi: start_bi.clone(),
            end_bi: end_bi.clone(),
            is_sure,
            dir,
            zs_lst: Vec::new(),
            eigen_fx: None,
            seg_idx: None,
            parent_seg: None,
            pre: None,
            next: None,
            bsp: None,
            bi_list: Vec::new(),
            reason: reason.to_string(),
            support_trend_line: None,
            resistance_trend_line: None,
            ele_inside_is_sure: false,
        };
        if end_bi.idx - start_bi.idx < 2 {
            seg.is_sure = false;
        }
        seg.check()?;
        Ok(seg)
    }

    // 其他方法省略，因为它们不直接用于 CZS
}
*/
#[derive(Debug, Clone)]
pub struct CZS<T: Clone> {
    pub is_sure: bool,
    pub sub_zs_lst: Vec<CZS<T>>,
    pub begin: CKLine_Unit,
    pub begin_bi: T,
    pub low: f64,
    pub high: f64,
    pub mid: f64,
    pub end: CKLine_Unit,
    pub end_bi: T,
    pub peak_high: f64,
    pub peak_low: f64,
    pub bi_in: Option<T>,
    pub bi_out: Option<T>,
    pub bi_lst: Vec<T>,
    pub _memoize_cache: HashMap<String, T>,
}

impl<T: Clone + PartialEq> CZS<T> {
    pub fn new(lst: Option<Vec<T>>, is_sure: bool) -> Self {
        let mut zs = Self {
            is_sure,
            sub_zs_lst: Vec::new(),
            begin: CKLine_Unit {
                time: 0,
                high: 0.0,
                low: 0.0,
            },
            begin_bi: lst.as_ref().unwrap()[0].clone(),
            low: 0.0,
            high: 0.0,
            mid: 0.0,
            end: CKLine_Unit {
                time: 0,
                high: 0.0,
                low: 0.0,
            },
            end_bi: lst.as_ref().unwrap()[0].clone(),
            peak_high: f64::NEG_INFINITY,
            peak_low: f64::INFINITY,
            bi_in: None,
            bi_out: None,
            bi_lst: Vec::new(),
            _memoize_cache: HashMap::new(),
        };

        if let Some(lst) = lst {
            zs.begin = lst[0].get_begin_klu();
            zs.begin_bi = lst[0].clone();
            zs.update_zs_range(&lst);

            for item in lst {
                zs.update_zs_end(item.clone());
            }
        }

        zs
    }

    pub fn clean_cache(&mut self) {
        self._memoize_cache.clear();
    }

    pub fn is_sure(&self) -> bool {
        self.is_sure
    }

    pub fn sub_zs_lst(&self) -> &Vec<CZS<T>> {
        &self.sub_zs_lst
    }

    pub fn begin(&self) -> &CKLine_Unit {
        &self.begin
    }

    pub fn begin_bi(&self) -> &T {
        &self.begin_bi
    }

    pub fn low(&self) -> f64 {
        self.low
    }

    pub fn high(&self) -> f64 {
        self.high
    }

    pub fn mid(&self) -> f64 {
        self.mid
    }

    pub fn end(&self) -> &CKLine_Unit {
        &self.end
    }

    pub fn end_bi(&self) -> &T {
        &self.end_bi
    }

    pub fn peak_high(&self) -> f64 {
        self.peak_high
    }

    pub fn peak_low(&self) -> f64 {
        self.peak_low
    }

    pub fn bi_in(&self) -> Option<&T> {
        self.bi_in.as_ref()
    }

    pub fn bi_out(&self) -> Option<&T> {
        self.bi_out.as_ref()
    }

    pub fn bi_lst(&self) -> &Vec<T> {
        &self.bi_lst
    }

    pub fn update_zs_range(&mut self, lst: &Vec<T>) {
        self.low = lst
            .iter()
            .map(|bi| bi._low())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        self.high = lst
            .iter()
            .map(|bi| bi._high())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        self.mid = (self.low + self.high) / 2.0;
        self.clean_cache();
    }

    pub fn is_one_bi_zs(&self) -> bool {
        assert!(self.end_bi.is_some());
        self.begin_bi.idx == self.end_bi.idx
    }

    pub fn update_zs_end(&mut self, item: T) {
        self.end = item.get_end_klu();
        self.end_bi = item.clone();
        if item._low() < self.peak_low {
            self.peak_low = item._low();
        }
        if item._high() > self.peak_high {
            self.peak_high = item._high();
        }
        self.clean_cache();
    }

    pub fn __str__(&self) -> String {
        let _str = format!("{}->{}", self.begin_bi.idx, self.end_bi.idx);
        let _str2 = self
            .sub_zs_lst
            .iter()
            .map(|sub_zs| sub_zs.__str__())
            .collect::<Vec<String>>()
            .join(",");
        if _str2.is_empty() {
            _str
        } else {
            format!("{}({})", _str, _str2)
        }
    }

    pub fn combine(&mut self, zs2: &CZS<T>, combine_mode: &str) -> bool {
        if zs2.is_one_bi_zs() {
            return false;
        }
        if self.begin_bi.seg_idx != zs2.begin_bi.seg_idx {
            return false;
        }
        match combine_mode {
            "zs" => {
                if !has_overlap(self.low, self.high, zs2.low, zs2.high, true) {
                    return false;
                }
                self.do_combine(zs2);
                true
            }
            "peak" => {
                if has_overlap(self.peak_low, self.peak_high, zs2.peak_low, zs2.peak_high) {
                    self.do_combine(zs2);
                    true
                } else {
                    false
                }
            }
            _ => {
                panic!("{} is unsupport zs conbine mode", combine_mode);
            }
        }
    }

    pub fn do_combine(&mut self, zs2: &CZS<T>) {
        if self.sub_zs_lst.is_empty() {
            self.sub_zs_lst.push(self.make_copy());
        }
        self.sub_zs_lst.push(zs2.clone());

        self.low = self.low.min(zs2.low);
        self.high = self.high.max(zs2.high);
        self.peak_low = self.peak_low.min(zs2.peak_low);
        self.peak_high = self.peak_high.max(zs2.peak_high);
        self.end = zs2.end.clone();
        self.bi_out = zs2.bi_out.clone();
        self.end_bi = zs2.end_bi.clone();
        self.clean_cache();
    }

    pub fn try_add_to_end(&mut self, item: T) -> bool {
        if !self.in_range(&item) {
            return false;
        }
        if self.is_one_bi_zs() {
            self.update_zs_range(&vec![self.begin_bi.clone(), item.clone()]);
        }
        self.update_zs_end(item);
        true
    }

    pub fn in_range(&self, item: &T) -> bool {
        has_overlap(self.low, self.high, item._low(), item._high())
    }

    pub fn is_inside(&self, seg: &CSeg<T>) -> bool {
        seg.start_bi.idx <= self.begin_bi.idx && self.begin_bi.idx <= seg.end_bi.idx
    }

    pub fn is_divergence(&self, config: &CPointConfig, out_bi: Option<&T>) -> (bool, Option<f64>) {
        if !self.end_bi_break(out_bi) {
            return (false, None);
        }
        let in_metric = self.get_bi_in().cal_macd_metric(config.macd_algo, false);
        let out_metric = if let Some(out_bi) = out_bi {
            out_bi.cal_macd_metric(config.macd_algo, true)
        } else {
            self.get_bi_out().cal_macd_metric(config.macd_algo, true)
        };

        if config.divergence_rate > 100 {
            (true, Some(out_metric / in_metric))
        } else {
            (
                out_metric <= config.divergence_rate * in_metric,
                Some(out_metric / in_metric),
            )
        }
    }

    pub fn init_from_zs(&mut self, zs: &CZS<T>) {
        self.begin = zs.begin.clone();
        self.end = zs.end.clone();
        self.low = zs.low;
        self.high = zs.high;
        self.peak_high = zs.peak_high;
        self.peak_low = zs.peak_low;
        self.begin_bi = zs.begin_bi.clone();
        self.end_bi = zs.end_bi.clone();
        self.bi_in = zs.bi_in.clone();
        self.bi_out = zs.bi_out.clone();
    }

    pub fn make_copy(&self) -> CZS<T> {
        let mut copy = CZS::new(None, self.is_sure);
        copy.init_from_zs(self);
        copy
    }

    pub fn end_bi_break(&self, end_bi: Option<&T>) -> bool {
        let end_bi = end_bi.unwrap_or_else(|| self.get_bi_out());
        assert!(end_bi.is_some());
        (end_bi.is_down() && end_bi._low() < self.low)
            || (end_bi.is_up() && end_bi._high() > self.high)
    }

    pub fn out_bi_is_peak(&self, end_bi_idx: i64) -> (bool, Option<f64>) {
        assert!(!self.bi_lst.is_empty());
        if self.bi_out.is_none() {
            return (false, None);
        }
        let bi_out = self.bi_out.as_ref().unwrap();
        let mut peak_rate = f64::INFINITY;
        for bi in &self.bi_lst {
            if bi.idx > end_bi_idx {
                break;
            }
            if (bi_out.is_down() && bi._low() < bi_out._low())
                || (bi_out.is_up() && bi._high() > bi_out._high())
            {
                return (false, None);
            }
            let r = (bi.get_end_val() - bi_out.get_end_val()).abs() / bi_out.get_end_val();
            if r < peak_rate {
                peak_rate = r;
            }
        }
        (true, Some(peak_rate))
    }

    pub fn get_bi_in(&self) -> &T {
        assert!(self.bi_in.is_some());
        self.bi_in.as_ref().unwrap()
    }

    pub fn get_bi_out(&self) -> &T {
        assert!(self.bi_out.is_some());
        self.bi_out.as_ref().unwrap()
    }

    pub fn set_bi_in(&mut self, bi: T) {
        self.bi_in = Some(bi);
        self.clean_cache();
    }

    pub fn set_bi_out(&mut self, bi: T) {
        self.bi_out = Some(bi);
        self.clean_cache();
    }

    pub fn set_bi_lst(&mut self, bi_lst: Vec<T>) {
        self.bi_lst = bi_lst;
        self.clean_cache();
    }
}

#[derive(Debug, Clone)]
pub struct CPointConfig {
    pub macd_algo: i32,
    pub divergence_rate: f64,
}

pub fn has_overlap(low1: f64, high1: f64, low2: f64, high2: f64, equal: bool) -> bool {
    if equal {
        low1 <= high2 && high1 >= low2
    } else {
        low1 < high2 && high1 > low2
    }
}
