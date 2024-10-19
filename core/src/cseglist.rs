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

    // 其他方法省略，因为它们不直接用于 CSegListChan
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

    // 其他方法省略，因为它们不直接用于 CSegListChan
}

#[derive(Debug, Clone)]
struct CZS<T: Clone> {
    // 省略具体实现，因为它们不直接用于 CSegListChan
}

impl<T: Clone> CZS<T> {
    fn is_one_bi_zs(&self) -> bool {
        // 省略具体实现
        false
    }
}

#[derive(Debug, Clone)]
struct CBS_Point {
    // 省略具体实现，因为它们不直接用于 CSegListChan
}

#[derive(Debug, Clone)]
struct CTrendLine {
    // 省略具体实现，因为它们不直接用于 CSegListChan
}

impl CTrendLine {
    fn new(bi_list: &Vec<CBi>, side: i32) -> Self {
        // 省略具体实现
        Self {}
    }
}

#[derive(Debug, Clone)]
struct CSegConfig {
    left_method: i32,
}

impl CSegConfig {
    fn new() -> Self {
        Self { left_method: 0 }
    }
}

#[derive(Debug, Clone)]
struct CSegListChan<T: Clone> {
    lst: Vec<CSeg<T>>,
    lv: i32,
    config: CSegConfig,
}

impl<T: Clone + PartialEq> CSegListChan<T> {
    fn new(seg_config: CSegConfig, lv: i32) -> Self {
        let mut seg_list = Self {
            lst: Vec::new(),
            lv,
            config: seg_config,
        };
        seg_list.do_init();
        seg_list
    }

    fn do_init(&mut self) {
        while let Some(seg) = self.lst.last() {
            if !seg.is_sure {
                for bi in &seg.bi_list {
                    bi.parent_seg = None;
                }
                if let Some(pre) = seg.pre.as_ref() {
                    pre.next = None;
                }
                self.lst.pop();
            } else {
                break;
            }
        }
        if let Some(seg) = self.lst.last() {
            if let Some(eigen_fx) = &seg.eigen_fx {
                if let Some(ele) = eigen_fx.ele.last() {
                    if let Some(lst) = ele.as_ref() {
                        if !lst.lst.last().unwrap().is_sure {
                            self.lst.pop();
                        }
                    }
                }
            }
        }
    }

    fn __iter__(&self) -> std::slice::Iter<CSeg<T>> {
        self.lst.iter()
    }

    fn __getitem__(&self, index: usize) -> &CSeg<T> {
        &self.lst[index]
    }

    fn __len__(&self) -> usize {
        self.lst.len()
    }

    fn left_bi_break(&self, bi_lst: &Vec<CBi>) -> bool {
        if self.lst.is_empty() {
            return false;
        }
        let last_seg_end_bi = &self.lst.last().unwrap().end_bi;
        for bi in bi_lst.iter().skip(last_seg_end_bi.idx as usize + 1) {
            if last_seg_end_bi.is_up() && bi._high() > last_seg_end_bi._high() {
                return true;
            } else if last_seg_end_bi.is_down() && bi._low() < last_seg_end_bi._low() {
                return true;
            }
        }
        false
    }

    fn collect_first_seg(&mut self, bi_lst: &Vec<CBi>) {
        if bi_lst.len() < 3 {
            return;
        }
        if self.config.left_method == 1 {
            // LEFT_SEG_METHOD.PEAK
            let _high = bi_lst
                .iter()
                .map(|bi| bi._high())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let _low = bi_lst
                .iter()
                .map(|bi| bi._low())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            if (_high - bi_lst[0].get_begin_val()).abs() >= (_low - bi_lst[0].get_begin_val()).abs()
            {
                if let Some(peak_bi) = find_peak_bi(bi_lst, true) {
                    self.add_new_seg(
                        bi_lst,
                        peak_bi.idx as usize,
                        false,
                        Some(1),
                        false,
                        "0seg_find_high",
                    ); // BI_DIR.UP
                }
            } else {
                if let Some(peak_bi) = find_peak_bi(bi_lst, false) {
                    self.add_new_seg(
                        bi_lst,
                        peak_bi.idx as usize,
                        false,
                        Some(-1),
                        false,
                        "0seg_find_low",
                    ); // BI_DIR.DOWN
                }
            }
            self.collect_left_as_seg(bi_lst);
        } else if self.config.left_method == 2 {
            // LEFT_SEG_METHOD.ALL
            let _dir = if bi_lst.last().unwrap().get_end_val() >= bi_lst[0].get_begin_val() {
                1
            } else {
                -1
            }; // BI_DIR.UP else BI_DIR.DOWN
            self.add_new_seg(
                bi_lst,
                bi_lst.len() - 1,
                false,
                Some(_dir),
                false,
                "0seg_collect_all",
            );
        } else {
            panic!("unknown seg left_method = {}", self.config.left_method);
        }
    }

    fn collect_left_seg_peak_method(&mut self, last_seg_end_bi: &CBi, bi_lst: &Vec<CBi>) {
        if last_seg_end_bi.is_down() {
            if let Some(peak_bi) =
                find_peak_bi(&bi_lst[last_seg_end_bi.idx as usize + 3..].to_vec(), true)
            {
                if peak_bi.idx - last_seg_end_bi.idx >= 3 {
                    self.add_new_seg(
                        bi_lst,
                        peak_bi.idx as usize,
                        false,
                        Some(1),
                        "collectleft_find_high",
                    ); // BI_DIR.UP
                }
            }
        } else {
            if let Some(peak_bi) =
                find_peak_bi(&bi_lst[last_seg_end_bi.idx as usize + 3..].to_vec(), false)
            {
                if peak_bi.idx - last_seg_end_bi.idx >= 3 {
                    self.add_new_seg(
                        bi_lst,
                        peak_bi.idx as usize,
                        false,
                        Some(-1),
                        "collectleft_find_low",
                    ); // BI_DIR.DOWN
                }
            }
        }
        let last_seg_end_bi = &self.lst.last().unwrap().end_bi;
        self.collect_left_as_seg(bi_lst);
    }

    fn collect_segs(&mut self, bi_lst: &Vec<CBi>) {
        let last_bi = bi_lst.last().unwrap();
        let last_seg_end_bi = &self.lst.last().unwrap().end_bi;
        if last_bi.idx - last_seg_end_bi.idx < 3 {
            return;
        }
        if last_seg_end_bi.is_down() && last_bi.get_end_val() <= last_seg_end_bi.get_end_val() {
            if let Some(peak_bi) =
                find_peak_bi(&bi_lst[last_seg_end_bi.idx as usize + 3..].to_vec(), true)
            {
                self.add_new_seg(
                    bi_lst,
                    peak_bi.idx as usize,
                    false,
                    Some(1),
                    "collectleft_find_high_force",
                ); // BI_DIR.UP
                self.collect_left_seg(bi_lst);
            }
        } else if last_seg_end_bi.is_up() && last_bi.get_end_val() >= last_seg_end_bi.get_end_val()
        {
            if let Some(peak_bi) =
                find_peak_bi(&bi_lst[last_seg_end_bi.idx as usize + 3..].to_vec(), false)
            {
                self.add_new_seg(
                    bi_lst,
                    peak_bi.idx as usize,
                    false,
                    Some(-1),
                    "collectleft_find_low_force",
                ); // BI_DIR.DOWN
                self.collect_left_seg(bi_lst);
            }
        } else if self.config.left_method == 2 {
            // LEFT_SEG_METHOD.ALL
            self.collect_left_as_seg(bi_lst);
        } else if self.config.left_method == 1 {
            // LEFT_SEG_METHOD.PEAK
            self.collect_left_seg_peak_method(last_seg_end_bi, bi_lst);
        } else {
            panic!("unknown seg left_method = {}", self.config.left_method);
        }
    }

    fn collect_left_seg(&mut self, bi_lst: &Vec<CBi>) {
        if self.lst.is_empty() {
            self.collect_first_seg(bi_lst);
        } else {
            self.collect_segs(bi_lst);
        }
    }

    fn collect_left_as_seg(&mut self, bi_lst: &Vec<CBi>) {
        let last_bi = bi_lst.last().unwrap();
        let last_seg_end_bi = &self.lst.last().unwrap().end_bi;
        if last_seg_end_bi.idx + 1 >= bi_lst.len() as i64 {
            return;
        }
        if last_seg_end_bi.dir == last_bi.dir {
            self.add_new_seg(
                bi_lst,
                last_bi.idx as usize - 1,
                false,
                None,
                "collect_left_1",
            );
        } else {
            self.add_new_seg(bi_lst, last_bi.idx as usize, false, None, "collect_left_0");
        }
    }

    fn try_add_new_seg(
        &mut self,
        bi_lst: &Vec<CBi>,
        end_bi_idx: usize,
        is_sure: bool,
        seg_dir: Option<i32>,
        split_first_seg: bool,
        reason: &str,
    ) {
        if self.lst.is_empty() && split_first_seg && end_bi_idx >= 3 {
            if let Some(peak_bi) = find_peak_bi(
                &bi_lst[end_bi_idx - 3..end_bi_idx].to_vec(),
                bi_lst[end_bi_idx].is_down(),
            ) {
                if (peak_bi.is_down() && (peak_bi._low() < bi_lst[0]._low() || peak_bi.idx == 0))
                    || (peak_bi.is_up()
                        && (peak_bi._high() > bi_lst[0]._high() || peak_bi.idx == 0))
                {
                    self.add_new_seg(
                        bi_lst,
                        peak_bi.idx as usize,
                        false,
                        Some(peak_bi.dir),
                        "split_first_1st",
                    );
                    self.add_new_seg(bi_lst, end_bi_idx, false, None, "split_first_2nd");
                    return;
                }
            }
        }
        let bi1_idx = if self.lst.is_empty() {
            0
        } else {
            self.lst.last().unwrap().end_bi.idx as usize + 1
        };
        let bi1 = &bi_lst[bi1_idx];
        let bi2 = &bi_lst[end_bi_idx];
        self.lst.push(
            CSeg::new(
                self.lst.len() as i64,
                bi1.clone(),
                bi2.clone(),
                is_sure,
                seg_dir,
                reason,
            )
            .unwrap(),
        );

        if self.lst.len() >= 2 {
            self.lst[self.lst.len() - 2].next =
                Some(Box::new(self.lst[self.lst.len() - 1].clone()));
            self.lst[self.lst.len() - 1].pre = Some(Box::new(self.lst[self.lst.len() - 2].clone()));
        }
        self.lst
            .last_mut()
            .unwrap()
            .update_bi_list(bi_lst, bi1_idx, end_bi_idx);
    }

    fn add_new_seg(
        &mut self,
        bi_lst: &Vec<CBi>,
        end_bi_idx: usize,
        is_sure: bool,
        seg_dir: Option<i32>,
        split_first_seg: bool,
        reason: &str,
    ) -> bool {
        match self.try_add_new_seg(
            bi_lst,
            end_bi_idx,
            is_sure,
            seg_dir,
            split_first_seg,
            reason,
        ) {
            Ok(_) => true,
            Err(e) => {
                if e.err_code == 1 && self.lst.is_empty() {
                    // ErrCode.SEG_END_VALUE_ERR
                    false
                } else {
                    panic!("{}", e.message);
                }
            }
        }
    }

    fn update(&mut self, bi_lst: &Vec<CBi>) {
        self.do_init();
        if self.lst.is_empty() {
            self.cal_seg_sure(bi_lst, 0);
        } else {
            self.cal_seg_sure(bi_lst, self.lst.last().unwrap().end_bi.idx as usize + 1);
        }
        self.collect_left_seg(bi_lst);
    }

    fn cal_seg_sure(&mut self, bi_lst: &Vec<CBi>, begin_idx: usize) {
        let mut up_eigen = CEigenFX::new(1, false, self.lv); // BI_DIR.UP
        let mut down_eigen = CEigenFX::new(-1, false, self.lv); // BI_DIR.DOWN
        let last_seg_dir = if self.lst.is_empty() {
            None
        } else {
            Some(self.lst.last().unwrap().dir)
        };
        for bi in bi_lst[begin_idx..].iter() {
            let mut fx_eigen = None;
            if bi.is_down() && last_seg_dir != Some(1) {
                if up_eigen.add(bi.clone()) {
                    fx_eigen = Some(up_eigen.clone());
                }
            } else if bi.is_up() && last_seg_dir != Some(-1) {
                if down_eigen.add(bi.clone()) {
                    fx_eigen = Some(down_eigen.clone());
                }
            }
            if self.lst.is_empty() {
                if up_eigen.ele[1].is_some() && bi.is_down() {
                    last_seg_dir = Some(-1); // BI_DIR.DOWN
                    down_eigen.clear();
                } else if down_eigen.ele[1].is_some() && bi.is_up() {
                    up_eigen.clear();
                    last_seg_dir = Some(1); // BI_DIR.UP
                }
                if up_eigen.ele[1].is_none() && last_seg_dir == Some(-1) && bi.dir == -1 {
                    last_seg_dir = None;
                } else if down_eigen.ele[1].is_none() && last_seg_dir == Some(1) && bi.dir == 1 {
                    last_seg_dir = None;
                }
            }
            if let Some(fx_eigen) = fx_eigen {
                self.treat_fx_eigen(fx_eigen, bi_lst);
                break;
            }
        }
    }

    fn treat_fx_eigen(&mut self, fx_eigen: CEigenFX, bi_lst: &Vec<CBi>) {
        let _test = fx_eigen.can_be_end(bi_lst);
        let end_bi_idx = fx_eigen.get_peak_bi_idx();
        if _test == Some(true) || _test == None {
            let is_true = _test != None;
            if !self.add_new_seg(
                bi_lst,
                end_bi_idx as usize,
                is_true && fx_eigen.all_bi_is_sure(),
                None,
                "normal",
            ) {
                self.cal_seg_sure(bi_lst, end_bi_idx as usize + 1);
                return;
            }
            self.lst.last_mut().unwrap().eigen_fx = Some(fx_eigen);
            if is_true {
                self.cal_seg_sure(bi_lst, end_bi_idx as usize + 1);
            }
        } else {
            self.cal_seg_sure(bi_lst, fx_eigen.lst[1].idx as usize);
        }
    }

    fn exist_sure_seg(&self) -> bool {
        self.lst.iter().any(|seg| seg.is_sure)
    }
}

fn find_peak_bi(bi_lst: &Vec<CBi>, is_high: bool) -> Option<CBi> {
    let mut peak_val = if is_high {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut peak_bi = None;
    for bi in bi_lst {
        if (is_high && bi.get_end_val() >= peak_val && bi.is_up())
            || (!is_high && bi.get_end_val() <= peak_val && bi.is_down())
        {
            if let Some(pre) = bi.next.as_ref() {
                if let Some(pre_pre) = pre.next.as_ref() {
                    if (is_high && pre_pre.get_end_val() > bi.get_end_val())
                        || (!is_high && pre_pre.get_end_val() < bi.get_end_val())
                    {
                        continue;
                    }
                }
            }
            peak_val = bi.get_end_val();
            peak_bi = Some(bi.clone());
        }
    }
    peak_bi
}
