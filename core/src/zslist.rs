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

    // 其他方法省略，因为它们不直接用于 CZSList
}

#[derive(Debug, Clone)]
struct CZS<T: Clone> {
    is_sure: bool,
    sub_zs_lst: Vec<CZS<T>>,
    begin: CKLineUnit,
    begin_bi: T,
    low: f64,
    high: f64,
    mid: f64,
    end: CKLineUnit,
    end_bi: T,
    peak_high: f64,
    peak_low: f64,
    bi_in: Option<T>,
    bi_out: Option<T>,
    bi_lst: Vec<T>,
    _memoize_cache: HashMap<String, T>,
}

impl<T: Clone + PartialEq> CZS<T> {
    fn new(lst: Option<Vec<T>>, is_sure: bool) -> Self {
        let mut zs = Self {
            is_sure,
            sub_zs_lst: Vec::new(),
            begin: CKLineUnit {
                time: 0,
                high: 0.0,
                low: 0.0,
            },
            begin_bi: lst.as_ref().unwrap()[0].clone(),
            low: 0.0,
            high: 0.0,
            mid: 0.0,
            end: CKLineUnit {
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

    // 其他方法省略，因为它们不直接用于 CZSList
}

#[derive(Debug, Clone)]
struct CZSConfig {
    one_bi_zs: bool,
    zs_algo: String,
    need_combine: bool,
    zs_combine_mode: String,
}

impl CZSConfig {
    fn new() -> Self {
        Self {
            one_bi_zs: false,
            zs_algo: "normal".to_string(),
            need_combine: false,
            zs_combine_mode: "peak".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct CZSList {
    zs_lst: Vec<CZS<CBi>>,
    config: CZSConfig,
    free_item_lst: Vec<CBi>,
    last_sure_pos: i64,
}

impl CZSList {
    fn new(zs_config: CZSConfig) -> Self {
        Self {
            zs_lst: Vec::new(),
            config: zs_config,
            free_item_lst: Vec::new(),
            last_sure_pos: -1,
        }
    }

    fn update_last_pos(&mut self, seg_list: &Vec<CSeg<CBi>>) {
        self.last_sure_pos = -1;
        for seg in seg_list.iter().rev() {
            if seg.is_sure {
                self.last_sure_pos = seg.start_bi.idx;
                return;
            }
        }
    }

    fn seg_need_cal(&self, seg: &CSeg<CBi>) -> bool {
        seg.start_bi.idx >= self.last_sure_pos
    }

    fn add_to_free_lst(&mut self, item: CBi, is_sure: bool, zs_algo: &str) {
        if !self.free_item_lst.is_empty() && item.idx == self.free_item_lst.last().unwrap().idx {
            self.free_item_lst.pop();
        }
        self.free_item_lst.push(item);
        if let Some(res) = self.try_construct_zs(&self.free_item_lst, is_sure, zs_algo) {
            if res.begin_bi.idx > 0 {
                self.zs_lst.push(res);
                self.clear_free_lst();
                self.try_combine();
            }
        }
    }

    fn clear_free_lst(&mut self) {
        self.free_item_lst.clear();
    }

    fn update(&mut self, bi: CBi, is_sure: bool) {
        if self.free_item_lst.is_empty() && self.try_add_to_end(bi) {
            self.try_combine();
            return;
        }
        self.add_to_free_lst(bi, is_sure, "normal");
    }

    fn try_add_to_end(&self, bi: CBi) -> bool {
        if self.zs_lst.is_empty() {
            false
        } else {
            self.zs_lst.last().unwrap().try_add_to_end(bi)
        }
    }

    fn add_zs_from_bi_range(&mut self, seg_bi_lst: Vec<CBi>, seg_dir: i32, seg_is_sure: bool) {
        let mut deal_bi_cnt = 0;
        for bi in seg_bi_lst {
            if bi.dir == seg_dir {
                continue;
            }
            if deal_bi_cnt < 1 {
                self.add_to_free_lst(bi, seg_is_sure, "normal");
                deal_bi_cnt += 1;
            } else {
                self.update(bi, seg_is_sure);
            }
        }
    }

    fn try_construct_zs(&self, lst: &Vec<CBi>, is_sure: bool, zs_algo: &str) -> Option<CZS<CBi>> {
        let mut lst = lst.clone();
        if zs_algo == "normal" {
            if !self.config.one_bi_zs {
                if lst.len() == 1 {
                    return None;
                } else {
                    lst = lst[lst.len() - 2..].to_vec();
                }
            }
        } else if zs_algo == "over_seg" {
            if lst.len() < 3 {
                return None;
            }
            lst = lst[lst.len() - 3..].to_vec();
            if lst[0].dir == lst[0].parent_seg.dir {
                lst = lst[1..].to_vec();
                return None;
            }
        }
        let min_high = lst
            .iter()
            .map(|item| item._high())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_low = lst
            .iter()
            .map(|item| item._low())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        if min_high > max_low {
            Some(CZS::new(Some(lst), is_sure))
        } else {
            None
        }
    }

    fn cal_bi_zs(&mut self, bi_lst: &Vec<CBi>, seg_lst: &Vec<CSeg<CBi>>) {
        while !self.zs_lst.is_empty()
            && self.zs_lst.last().unwrap().begin_bi.idx >= self.last_sure_pos
        {
            self.zs_lst.pop();
        }
        if self.config.zs_algo == "normal" {
            for seg in seg_lst {
                if !self.seg_need_cal(seg) {
                    continue;
                }
                self.clear_free_lst();
                let seg_bi_lst =
                    bi_lst[seg.start_bi.idx as usize..=seg.end_bi.idx as usize].to_vec();
                self.add_zs_from_bi_range(seg_bi_lst, seg.dir, seg.is_sure);
            }
            if !seg_lst.is_empty() {
                self.clear_free_lst();
                self.add_zs_from_bi_range(
                    bi_lst[seg_lst.last().unwrap().end_bi.idx as usize + 1..].to_vec(),
                    revert_bi_dir(seg_lst.last().unwrap().dir),
                    false,
                );
            }
        } else if self.config.zs_algo == "over_seg" {
            assert!(!self.config.one_bi_zs);
            self.clear_free_lst();
            let begin_bi_idx = if !self.zs_lst.is_empty() {
                self.zs_lst.last().unwrap().end_bi.idx + 1
            } else {
                0
            };
            for bi in bi_lst[begin_bi_idx as usize..].iter() {
                self.update_overseg_zs(bi.clone());
            }
        } else if self.config.zs_algo == "auto" {
            let mut sure_seg_appear = false;
            let exist_sure_seg = seg_lst.iter().any(|seg| seg.is_sure);
            for seg in seg_lst {
                if seg.is_sure {
                    sure_seg_appear = true;
                }
                if !self.seg_need_cal(seg) {
                    continue;
                }
                if seg.is_sure || (!sure_seg_appear && exist_sure_seg) {
                    self.clear_free_lst();
                    let seg_bi_lst =
                        bi_lst[seg.start_bi.idx as usize..=seg.end_bi.idx as usize].to_vec();
                    self.add_zs_from_bi_range(seg_bi_lst, seg.dir, seg.is_sure);
                } else {
                    self.clear_free_lst();
                    for bi in bi_lst[seg.start_bi.idx as usize..].iter() {
                        self.update_overseg_zs(bi.clone());
                    }
                    break;
                }
            }
        } else {
            panic!("unknown zs_algo {}", self.config.zs_algo);
        }
        self.update_last_pos(seg_lst);
    }

    fn update_overseg_zs(&mut self, bi: CBi) {
        if !self.zs_lst.is_empty() && self.free_item_lst.is_empty() {
            if bi.next.is_none() {
                return;
            }
            if bi.idx - self.zs_lst.last().unwrap().end_bi.idx <= 1
                && self
                    .zs_lst
                    .last()
                    .unwrap()
                    .in_range(bi.next.as_ref().unwrap())
                && self.zs_lst.last().unwrap().try_add_to_end(bi.clone())
            {
                return;
            }
        }
        if !self.zs_lst.is_empty()
            && self.free_item_lst.is_empty()
            && self.zs_lst.last().unwrap().in_range(&bi)
            && bi.idx - self.zs_lst.last().unwrap().end_bi.idx <= 1
        {
            return;
        }
        self.add_to_free_lst(bi, bi.is_sure, "over_seg");
    }

    fn __iter__(&self) -> std::slice::Iter<CZS<CBi>> {
        self.zs_lst.iter()
    }

    fn __len__(&self) -> usize {
        self.zs_lst.len()
    }

    fn __getitem__(&self, index: usize) -> &CZS<CBi> {
        &self.zs_lst[index]
    }

    fn try_combine(&mut self) {
        if !self.config.need_combine {
            return;
        }
        while self.zs_lst.len() >= 2
            && self.zs_lst[self.zs_lst.len() - 2]
                .combine(self.zs_lst.last().unwrap(), &self.config.zs_combine_mode)
        {
            self.zs_lst.pop();
        }
    }
}

fn revert_bi_dir(dir: i32) -> i32 {
    if dir == 1 {
        -1
    } else {
        1
    }
}
