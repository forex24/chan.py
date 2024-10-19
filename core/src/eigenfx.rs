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
}

#[derive(Debug, Clone)]
struct CKLineUnit {
    time: i64,
    high: f64,
    low: f64,
}

#[derive(Debug, Clone)]
struct CEigen {
    time_begin: i64,
    time_end: i64,
    high: f64,
    low: f64,
    lst: Vec<CBi>,
    dir: i32,
    fx: i32,
    pre: Option<Box<CEigen>>,
    next: Option<Box<CEigen>>,
    memoize_cache: HashMap<String, CBi>,
    gap: bool,
}

impl CEigen {
    fn new(bi: CBi, dir: i32) -> Result<Self, CChanException> {
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

    fn clean_cache(&mut self) {
        self.memoize_cache.clear();
    }

    fn test_combine(
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

    fn add(&mut self, bi: CBi) {
        self.lst.push(bi);
    }

    fn set_fx(&mut self, fx: i32) {
        self.fx = fx;
    }

    fn try_add(
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

    fn get_peak_klu(&self, is_high: bool) -> Result<CBi, CChanException> {
        if is_high {
            self.get_high_peak_klu()
        } else {
            self.get_low_peak_klu()
        }
    }

    fn get_high_peak_klu(&self) -> Result<CBi, CChanException> {
        for bi in self.lst.iter().rev() {
            if bi._high() == self.high {
                return Ok(bi.clone());
            }
        }
        Err(CChanException::new("can't find peak...".to_string(), 2)) // ErrCode.COMBINER_ERR
    }

    fn get_low_peak_klu(&self) -> Result<CBi, CChanException> {
        for bi in self.lst.iter().rev() {
            if bi._low() == self.low {
                return Ok(bi.clone());
            }
        }
        Err(CChanException::new("can't find peak...".to_string(), 2)) // ErrCode.COMBINER_ERR
    }

    fn update_fx(
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

    fn get_peak_bi_idx(&self) -> Result<i64, CChanException> {
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

    fn set_pre(&mut self, pre: Box<CEigen>) {
        self.pre = Some(pre);
        self.clean_cache();
    }

    fn set_next(&mut self, next: Box<CEigen>) {
        self.next = Some(next);
        self.clean_cache();
    }
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

    fn treat_first_ele(&mut self, bi: CBi) -> bool {
        self.ele[0] = Some(CEigen::new(bi, self.kl_dir).unwrap());
        false
    }

    fn treat_second_ele(&mut self, bi: CBi) -> bool {
        if let Some(ref mut ele0) = self.ele[0] {
            let combine_dir = ele0.try_add(bi, self.exclude_included, None).unwrap();
            if combine_dir != 0 {
                // KLINE_DIR.COMBINE
                self.ele[1] = Some(CEigen::new(bi, self.kl_dir).unwrap());
                if (self.is_up() && self.ele[1].as_ref().unwrap().high < ele0.high)
                    || (self.is_down() && self.ele[1].as_ref().unwrap().low > ele0.low)
                {
                    return self.reset();
                }
            }
        }
        false
    }

    fn treat_third_ele(&mut self, bi: CBi) -> bool {
        self.last_evidence_bi = Some(bi.clone());
        if let Some(ref mut ele1) = self.ele[1] {
            let allow_top_equal = if self.exclude_included {
                if bi.is_down() {
                    1
                } else {
                    -1
                }
            } else {
                None
            };
            let combine_dir = ele1
                .try_add(bi, self.exclude_included, allow_top_equal)
                .unwrap();
            if combine_dir == 0 {
                // KLINE_DIR.COMBINE
                return false;
            }
            self.ele[2] = Some(CEigen::new(bi, combine_dir).unwrap());
            if !self.actual_break() {
                return self.reset();
            }
            if let Some(ref mut ele2) = self.ele[2] {
                ele1.update_fx(
                    Box::new(self.ele[0].as_ref().unwrap().clone()),
                    Box::new(ele2.clone()),
                    self.exclude_included,
                    allow_top_equal,
                )
                .unwrap();
                let fx = ele1.fx;
                let is_fx = (self.is_up() && fx == 1) || (self.is_down() && fx == -1);
                return if is_fx { true } else { self.reset() };
            }
        }
        false
    }

    fn add(&mut self, bi: CBi) -> bool {
        assert_ne!(bi.dir, self.dir);
        self.lst.push(bi.clone());
        if self.ele[0].is_none() {
            // 第一元素
            return self.treat_first_ele(bi);
        } else if self.ele[1].is_none() {
            // 第二元素
            return self.treat_second_ele(bi);
        } else if self.ele[2].is_none() {
            // 第三元素
            return self.treat_third_ele(bi);
        } else {
            panic!(
                "特征序列3个都找齐了还没处理!! 当前笔:{},当前:{}",
                bi.idx, self
            );
        }
    }

    fn reset(&mut self) -> bool {
        let bi_tmp_list = self.lst[1..].to_vec();
        if self.exclude_included {
            self.clear();
            for bi in bi_tmp_list {
                if self.add(bi) {
                    return true;
                }
            }
        } else {
            if let Some(ref ele1) = self.ele[1] {
                let ele2_begin_idx = ele1.lst[0].idx;
                self.ele[0] = self.ele[1].take();
                self.ele[1] = self.ele[2].take();
                self.ele[2] = None;
                self.lst = bi_tmp_list
                    .into_iter()
                    .filter(|bi| bi.idx >= ele2_begin_idx)
                    .collect();
            }
        }
        false
    }

    fn can_be_end(&self, bi_lst: &Vec<CBi>) -> bool {
        if let Some(ref ele1) = self.ele[1] {
            if ele1.gap {
                if let Some(ref ele0) = self.ele[0] {
                    let end_bi_idx = self.get_peak_bi_idx();
                    let thred_value = bi_lst[end_bi_idx as usize].get_end_val();
                    let break_thred = if self.is_up() { ele0.low } else { ele0.high };
                    return self.find_revert_fx(bi_lst, end_bi_idx + 2, thred_value, break_thred);
                }
            } else {
                return true;
            }
        }
        false
    }

    fn is_down(&self) -> bool {
        self.dir == 1
    }

    fn is_up(&self) -> bool {
        self.dir == -1
    }

    fn get_peak_bi_idx(&self) -> i64 {
        self.ele[1].as_ref().unwrap().get_peak_bi_idx().unwrap()
    }

    fn all_bi_is_sure(&self) -> bool {
        if let Some(ref last_evidence_bi) = self.last_evidence_bi {
            self.lst.iter().all(|bi| bi.is_sure) && last_evidence_bi.is_sure
        } else {
            false
        }
    }

    fn clear(&mut self) {
        self.ele = vec![None, None, None];
        self.lst = Vec::new();
    }

    fn actual_break(&self) -> bool {
        if !self.exclude_included {
            return true;
        }
        if let (Some(ref ele2), Some(ref ele1)) = (&self.ele[2], &self.ele[1]) {
            if (self.is_up() && ele2.low < ele1.lst.last().unwrap()._low())
                || (self.is_down() && ele2.high > ele1.lst.last().unwrap()._high())
            {
                return true;
            }
            if ele2.lst.len() == 1 {
                let ele2_bi = &ele2.lst[0];
                if let (Some(ref next), Some(ref next_next)) = (
                    ele2_bi.next.as_ref(),
                    ele2_bi.next.as_ref().and_then(|n| n.next.as_ref()),
                ) {
                    if ele2_bi.is_down() && next_next._low() < ele2_bi._low() {
                        return true;
                    } else if ele2_bi.is_up() && next_next._high() > ele2_bi._high() {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn find_revert_fx(
        &self,
        bi_list: &Vec<CBi>,
        begin_idx: i64,
        thred_value: f64,
        break_thred: f64,
    ) -> bool {
        let first_bi_dir = bi_list[begin_idx as usize].dir;
        let egien_fx = CEigenFX::new(
            if first_bi_dir == 1 { -1 } else { 1 },
            !self.exclude_included,
            self.lv,
        );
        for bi in bi_list.iter().skip(begin_idx as usize).step_by(2) {
            if egien_fx.add(bi.clone()) {
                if !self.exclude_included {
                    return true;
                }
                while let Some(ref ele1) = egien_fx.ele[1] {
                    if egien_fx.can_be_end(bi_list) {
                        return true;
                    }
                    if !egien_fx.reset() {
                        break;
                    }
                }
            }
            if (bi.is_down() && bi._low() < thred_value) || (bi.is_up() && bi._high() > thred_value)
            {
                return false;
            }
            if let Some(ref ele1) = egien_fx.ele[1] {
                if (bi.is_down() && ele1.high > break_thred)
                    || (bi.is_up() && ele1.low < break_thred)
                {
                    return true;
                }
            }
        }
        false
    }
}
