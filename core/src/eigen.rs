use std::collections::HashMap;

use crate::{bi::CBi, cenum::CChanException};
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

#[derive(Debug, Clone)]
struct CBi {
    begin_klc: CKLineUnit,
    end_klc: CKLineUnit,
    high: f64,
    low: f64,
}

impl CBi {
    pub fn _high(&self) -> f64 {
        self.high
    }

    pub fn _low(&self) -> f64 {
        self.low
    }
}

#[derive(Debug, Clone)]
struct CKLineUnit {
    time: i64,
    high: f64,
    low: f64,
}

#[derive(Debug, Clone)]
struct CSeg {
    start_bi: CBi,
    end_bi: CBi,
    high: f64,
    low: f64,
}

impl CSeg {
    pub fn _high(&self) -> f64 {
        self.high
    }

    pub fn _low(&self) -> f64 {
        self.low
    }
}*/

#[derive(Debug, Clone)]
pub struct CEigen {
    pub time_begin: i64,
    pub time_end: i64,
    pub high: f64,
    pub low: f64,
    pub lst: Vec<CBi>,
    pub dir: i32,
    pub fx: i32,
    pub pre: Option<Box<CEigen>>,
    pub next: Option<Box<CEigen>>,
    pub memoize_cache: HashMap<String, CBi>,
    pub gap: bool,
}

impl CEigen {
    pub fn new(bi: CBi, dir: i32) -> Result<Self, CChanException> {
        let time_begin = bi.begin_klc.time;
        let time_end = bi.end_klc.time;
        let high = bi._high();
        let low = bi._low();
        Ok(Self {
            time_begin,
            time_end,
            high,
            low,
            lst: vec![bi],
            dir,
            fx: 0, // FX_TYPE.UNKNOWN
            pre: None,
            next: None,
            memoize_cache: HashMap::new(),
            gap: false,
        })
    }

    pub fn clean_cache(&mut self) {
        self.memoize_cache.clear();
    }

    pub fn test_combine(
        &self,
        bi: &CBi,
        exclude_included: bool,
        allow_top_equal: Option<i32>,
    ) -> Result<i32, CChanException> {
        let item_high = bi._high();
        let item_low = bi._low();
        if self.high >= item_high && self.low <= item_low {
            return Ok(0); // KLINE_DIR.COMBINE
        }
        if self.high <= item_high && self.low >= item_low {
            if allow_top_equal == Some(1) && self.high == item_high && self.low > item_low {
                return Ok(1); // KLINE_DIR.DOWN
            } else if allow_top_equal == Some(-1) && self.low == item_low && self.high < item_high {
                return Ok(-1); // KLINE_DIR.UP
            }
            return Ok(2); // KLINE_DIR.INCLUDED if exclude_included else KLINE_DIR.COMBINE
        }
        if self.high > item_high && self.low > item_low {
            return Ok(1); // KLINE_DIR.DOWN
        }
        if self.high < item_high && self.low < item_low {
            return Ok(-1); // KLINE_DIR.UP
        }
        Err(CChanException::new("combine type unknown".to_string(), 2)) // ErrCode.COMBINER_ERR
    }

    pub fn add(&mut self, bi: CBi) {
        self.lst.push(bi);
    }

    pub fn set_fx(&mut self, fx: i32) {
        self.fx = fx;
    }

    pub fn try_add(
        &mut self,
        bi: CBi,
        exclude_included: bool,
        allow_top_equal: Option<i32>,
    ) -> Result<i32, CChanException> {
        let dir = self.test_combine(&bi, exclude_included, allow_top_equal)?;
        if dir == 0 {
            // KLINE_DIR.COMBINE
            self.lst.push(bi);
            if self.dir == -1 {
                // KLINE_DIR.UP
                if bi._high() != bi._low() || bi._high() != self.high {
                    self.high = self.high.max(bi._high());
                    self.low = self.low.max(bi._low());
                }
            } else if self.dir == 1 {
                // KLINE_DIR.DOWN
                if bi._high() != bi._low() || bi._low() != self.low {
                    self.high = self.high.min(bi._high());
                    self.low = self.low.min(bi._low());
                }
            } else {
                return Err(CChanException::new(
                    format!("KLINE_DIR = {} err!!! must be -1/1", self.dir),
                    2, // ErrCode.COMBINER_ERR
                ));
            }
            self.time_end = bi.end_klc.time;
            self.clean_cache();
        }
        Ok(dir)
    }

    pub fn get_peak_klu(&self, is_high: bool) -> Result<CBi, CChanException> {
        if is_high {
            self.get_high_peak_klu()
        } else {
            self.get_low_peak_klu()
        }
    }

    pub fn get_high_peak_klu(&self) -> Result<CBi, CChanException> {
        for bi in self.lst.iter().rev() {
            if bi._high() == self.high {
                return Ok(bi.clone());
            }
        }
        Err(CChanException::new("can't find peak...".to_string(), 2)) // ErrCode.COMBINER_ERR
    }

    pub fn get_low_peak_klu(&self) -> Result<CBi, CChanException> {
        for bi in self.lst.iter().rev() {
            if bi._low() == self.low {
                return Ok(bi.clone());
            }
        }
        Err(CChanException::new("can't find peak...".to_string(), 2)) // ErrCode.COMBINER_ERR
    }

    pub fn update_fx(
        &mut self,
        pre: Box<CEigen>,
        next: Box<CEigen>,
        exclude_included: bool,
        allow_top_equal: Option<i32>,
    ) -> Result<(), CChanException> {
        self.set_next(next.clone());
        self.set_pre(pre.clone());
        let next_ref = next.as_ref();
        next_ref.set_pre(Box::new(self.clone()));

        if exclude_included {
            let pre_ref = pre.as_ref();
            if pre_ref.high < self.high && next_ref.high <= self.high && next_ref.low < self.low {
                if allow_top_equal == Some(1) || next_ref.high < self.high {
                    self.fx = 1; // FX_TYPE.TOP
                }
            } else if next_ref.high > self.high
                && pre_ref.low > self.low
                && next_ref.low >= self.low
            {
                if allow_top_equal == Some(-1) || next_ref.low > self.low {
                    self.fx = -1; // FX_TYPE.BOTTOM
                }
            }
        } else if pre_ref.high < self.high
            && next_ref.high < self.high
            && pre_ref.low < self.low
            && next_ref.low < self.low
        {
            self.fx = 1; // FX_TYPE.TOP
        } else if pre_ref.high > self.high
            && next_ref.high > self.high
            && pre_ref.low > self.low
            && next_ref.low > self.low
        {
            self.fx = -1; // FX_TYPE.BOTTOM
        }
        self.clean_cache();

        if (self.fx == 1 && pre_ref.high < self.low) || (self.fx == -1 && pre_ref.low > self.high) {
            self.gap = true;
        }
        Ok(())
    }

    pub fn get_peak_bi_idx(&self) -> Result<i64, CChanException> {
        if self.fx == 0 {
            // FX_TYPE.UNKNOWN
            return Err(CChanException::new("fx is UNKNOWN".to_string(), 1)); // ErrCode.COMMON_ERROR
        }
        let bi_dir = self.lst[0].dir;
        if bi_dir == -1 {
            // BI_DIR.UP
            Ok(self.get_peak_klu(false)?.idx - 1)
        } else {
            Ok(self.get_peak_klu(true)?.idx - 1)
        }
    }

    pub fn set_pre(&mut self, pre: Box<CEigen>) {
        self.pre = Some(pre);
        self.clean_cache();
    }

    pub fn set_next(&mut self, next: Box<CEigen>) {
        self.next = Some(next);
        self.clean_cache();
    }
}
