use std::collections::HashMap;

#[derive(Debug, Clone)]
struct CChanException {
    message: String,
    err_code: i32,
}

impl CChanException {
    fn new(message: String, err_code: i32) -> Self {
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
    fn _high(&self) -> f64 {
        self.high
    }

    fn _low(&self) -> f64 {
        self.low
    }

    fn is_down(&self) -> bool {
        self.dir == 1
    }

    fn is_up(&self) -> bool {
        self.dir == -1
    }

    fn get_end_val(&self) -> f64 {
        self.end_klc.high
    }

    fn get_begin_val(&self) -> f64 {
        self.begin_klc.high
    }

    fn get_end_klu(&self) -> CKLineUnit {
        self.end_klc.clone()
    }

    fn get_begin_klu(&self) -> CKLineUnit {
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
struct CEigenFX {
    lv: i32,
    dir: i32,
    ele: Vec<Option<CEigen>>,
    lst: Vec<CBi>,
    exclude_included: bool,
    kl_dir: i32,
    last_evidence_bi: Option<CBi>,
}

impl CEigenFX {
    fn new(dir: i32, exclude_included: bool, lv: i32) -> Self {
        Self {
            lv,
            dir,
            ele: vec![None, None, None],
            lst: Vec::new(),
            exclude_included,
            kl_dir: if dir == -1 { 1 } else { -1 }, // KLINE_DIR.UP if _dir == BI_DIR.UP else KLINE_DIR.DOWN
            last_evidence_bi: None,
        }
    }

    // 其他方法省略，因为它们不直接用于 CSeg
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
    fn new(
        idx: i64,
        start_bi: T,
        end_bi: T,
        is_sure: bool,
        seg_dir: Option<i32>,
        reason: &str,
    ) -> Result<Self, CChanException> {
        assert!(
            start_bi.idx == 0 || start_bi.dir == end_bi.dir || !is_sure,
            format!(
                "{} {} {} {}",
                start_bi.idx, end_bi.idx, start_bi.dir, end_bi.dir
            )
        );
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

    fn set_seg_idx(&mut self, idx: i64) {
        self.seg_idx = Some(idx);
    }

    fn check(&self) -> Result<(), CChanException> {
        if !self.is_sure {
            return Ok(());
        }
        if self.is_down() {
            if self.start_bi.get_begin_val() < self.end_bi.get_end_val() {
                return Err(CChanException::new(
                    format!("下降线段起始点应该高于结束点! idx={}", self.idx),
                    1,
                )); // ErrCode.SEG_END_VALUE_ERR
            }
        } else if self.start_bi.get_begin_val() > self.end_bi.get_end_val() {
            return Err(CChanException::new(
                format!("上升线段起始点应该低于结束点! idx={}", self.idx),
                1,
            )); // ErrCode.SEG_END_VALUE_ERR
        }
        if self.end_bi.idx - self.start_bi.idx < 2 {
            return Err(CChanException::new(
                format!(
                    "线段({}-{})长度不能小于2! idx={}",
                    self.start_bi.idx, self.end_bi.idx, self.idx
                ),
                2,
            )); // ErrCode.SEG_LEN_ERR
        }
        Ok(())
    }

    fn __str__(&self) -> String {
        format!(
            "{}->{}: {}  {}",
            self.start_bi.idx, self.end_bi.idx, self.dir, self.is_sure
        )
    }

    fn add_zs(&mut self, zs: CZS<T>) {
        self.zs_lst.insert(0, zs); // 因为中枢是反序加入的
    }

    fn cal_klu_slope(&self) -> f64 {
        assert!(self.end_bi.idx >= self.start_bi.idx);
        (self.get_end_val() - self.get_begin_val())
            / (self.get_end_klu().idx - self.get_begin_klu().idx)
            / self.get_begin_val()
    }

    fn cal_amp(&self) -> f64 {
        (self.get_end_val() - self.get_begin_val()) / self.get_begin_val()
    }

    fn cal_bi_cnt(&self) -> i64 {
        self.end_bi.idx - self.start_bi.idx + 1
    }

    fn clear_zs_lst(&mut self) {
        self.zs_lst.clear();
    }

    fn _low(&self) -> f64 {
        if self.is_down() {
            self.end_bi.get_end_klu().low
        } else {
            self.start_bi.get_begin_klu().low
        }
    }

    fn _high(&self) -> f64 {
        if self.is_up() {
            self.end_bi.get_end_klu().high
        } else {
            self.start_bi.get_begin_klu().high
        }
    }

    fn is_down(&self) -> bool {
        self.dir == 1
    }

    fn is_up(&self) -> bool {
        self.dir == -1
    }

    fn get_end_val(&self) -> f64 {
        self.end_bi.get_end_val()
    }

    fn get_begin_val(&self) -> f64 {
        self.start_bi.get_begin_val()
    }

    fn amp(&self) -> f64 {
        (self.get_end_val() - self.get_begin_val()).abs()
    }

    fn get_end_klu(&self) -> CKLineUnit {
        self.end_bi.get_end_klu()
    }

    fn get_begin_klu(&self) -> CKLineUnit {
        self.start_bi.get_begin_klu()
    }

    fn get_klu_cnt(&self) -> i64 {
        self.get_end_klu().idx - self.get_begin_klu().idx + 1
    }

    fn cal_macd_metric(&self, macd_algo: i32, is_reverse: bool) -> Result<f64, CChanException> {
        match macd_algo {
            1 => Ok(self.cal_macd_slope()), // MACD_ALGO.SLOPE
            2 => Ok(self.cal_macd_amp()),   // MACD_ALGO.AMP
            _ => Err(CChanException::new(
                format!(
                    "unsupport macd_algo={} of Seg, should be one of slope/amp",
                    macd_algo
                ),
                3,
            )), // ErrCode.PARA_ERROR
        }
    }

    fn cal_macd_slope(&self) -> f64 {
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

    fn cal_macd_amp(&self) -> f64 {
        let begin_klu = self.get_begin_klu();
        let end_klu = self.get_end_klu();
        if self.is_down() {
            (begin_klu.high - end_klu.low) / begin_klu.high
        } else {
            (end_klu.high - begin_klu.low) / begin_klu.low
        }
    }

    fn update_bi_list(&mut self, bi_lst: &Vec<T>, idx1: i64, idx2: i64) {
        for bi_idx in idx1..=idx2 {
            let bi = &bi_lst[bi_idx as usize];
            self.bi_list.push(bi.clone());
        }
        if self.bi_list.len() >= 3 {
            self.support_trend_line = Some(CTrendLine::new(&self.bi_list, 1)); // TREND_LINE_SIDE.INSIDE
            self.resistance_trend_line = Some(CTrendLine::new(&self.bi_list, 2));
            // TREND_LINE_SIDE.OUTSIDE
        }
    }

    fn get_first_multi_bi_zs(&self) -> Option<CZS<T>> {
        self.zs_lst.iter().find(|zs| !zs.is_one_bi_zs()).cloned()
    }

    fn get_final_multi_bi_zs(&self) -> Option<CZS<T>> {
        self.zs_lst
            .iter()
            .rev()
            .find(|zs| !zs.is_one_bi_zs())
            .cloned()
    }

    fn get_multi_bi_zs_cnt(&self) -> usize {
        self.zs_lst.iter().filter(|zs| !zs.is_one_bi_zs()).count()
    }
}

#[derive(Debug, Clone)]
struct CZS<T: Clone> {
    // 省略具体实现，因为它们不直接用于 CSeg
}

impl<T: Clone> CZS<T> {
    fn is_one_bi_zs(&self) -> bool {
        // 省略具体实现
        false
    }
}

#[derive(Debug, Clone)]
struct CBS_Point {
    // 省略具体实现，因为它们不直接用于 CSeg
}

#[derive(Debug, Clone)]
struct CTrendLine {
    // 省略具体实现，因为它们不直接用于 CSeg
}

impl CTrendLine {
    fn new(bi_list: &Vec<CBi>, side: i32) -> Self {
        // 省略具体实现
        Self {}
    }
}
