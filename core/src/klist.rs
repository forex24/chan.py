use std::collections::HashMap;
use std::fmt;

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

#[derive(Debug, Clone, PartialEq)]
enum FX_CHECK_METHOD {
    HALF,
    LOSS,
    STRICT,
    TOTALLY,
}

#[derive(Debug, Clone, PartialEq)]
enum FX_TYPE {
    TOP,
    BOTTOM,
    UNKNOWN,
}

#[derive(Debug, Clone, PartialEq)]
enum KLINE_DIR {
    UP,
    DOWN,
    COMBINE,
}

#[derive(Debug, Clone)]
struct CKLine_Unit {
    kl_type: Option<String>,
    time: CTime,
    close: f64,
    open: f64,
    high: f64,
    low: f64,
    trade_info: CTradeInfo,
    demark: CDemarkIndex,
    sub_kl_list: Vec<CKLine_Unit>,
    sup_kl: Option<Box<CKLine_Unit>>,
    klc: Option<CKLine>,
    trend: HashMap<TREND_TYPE, HashMap<i32, f64>>,
    limit_flag: i32,
    pre: Option<Box<CKLine_Unit>>,
    next: Option<Box<CKLine_Unit>>,
    idx: i32,
}

impl CKLine_Unit {
    fn new(kl_dict: HashMap<DATA_FIELD, f64>, autofix: bool) -> Self {
        let time = CTime { time: kl_dict[&DATA_FIELD::FIELD_TIME].to_string() };
        let close = kl_dict[&DATA_FIELD::FIELD_CLOSE];
        let open = kl_dict[&DATA_FIELD::FIELD_OPEN];
        let high = kl_dict[&DATA_FIELD::FIELD_HIGH];
        let low = kl_dict[&DATA_FIELD::FIELD_LOW];

        let mut klu = Self {
            kl_type: None,
            time,
            close,
            open,
            high,
            low,
            trade_info: CTradeInfo::new(kl_dict),
            demark: CDemarkIndex::new(),
            sub_kl_list: Vec::new(),
            sup_kl: None,
            klc: None,
            trend: HashMap::new(),
            limit_flag: 0,
            pre: None,
            next: None,
            idx: -1,
        };

        klu.check(autofix);
        klu
    }

    fn check(&mut self, autofix: bool) {
        let min_price = self.low.min(self.open).min(self.high).min(self.close);
        let max_price = self.low.max(self.open).max(self.high).max(self.close);

        if self.low > min_price {
            if autofix {
                self.low = min_price;
            } else {
                panic!("{} low price={} is not min of [low={}, open={}, high={}, close={}]", self.time.to_str(), self.low, self.low, self.open, self.high, self.close);
            }
        }

        if self.high < max_price {
            if autofix {
                self.high = max_price;
            } else {
                panic!("{} high price={} is not max of [low={}, open={}, high={}, close={}]", self.time.to_str(), self.high, self.low, self.open, self.high, self.close);
            }
        }
    }

    fn add_children(&mut self, child: CKLine_Unit) {
        self.sub_kl_list.push(child);
    }

    fn set_parent(&mut self, parent: CKLine_Unit) {
        self.sup_kl = Some(Box::new(parent));
    }

    fn get_children(&self) -> impl Iterator<Item = &CKLine_Unit> {
        self.sub_kl_list.iter()
    }

    fn _low(&self) -> f64 {
        self.low
    }

    fn _high(&self) -> f64 {
        self.high
    }

    fn set_metric(&mut self, metric_model_lst: Vec<Box<dyn MetricModel>>) {
        for metric_model in metric_model_lst {
            metric_model.add(self);
        }
    }

    fn get_parent_klc(&self) -> &CKLine {
        self.sup_kl.as_ref().unwrap().klc.as_ref().unwrap()
    }

    fn include_sub_lv_time(&self, sub_lv_t: &str) -> bool {
        if self.time.to_str() == sub_lv_t {
            return true;
        }
        for sub_klu in &self.sub_kl_list {
            if sub_klu.time.to_str() == sub_lv_t {
                return true;
            }
            if sub_klu.include_sub_lv_time(sub_lv_t) {
                return true;
            }
        }
        false
    }

    fn set_pre_klu(&mut self, pre_klu: Option<CKLine_Unit>) {
        if let Some(pre_klu) = pre_klu {
            self.pre = Some(Box::new(pre_klu.clone()));
            pre_klu.next = Some(Box::new(self.clone()));
        }
    }
}

impl fmt::Display for CKLine_Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} open={} close={} high={} low={} {}", self.idx, self.time.to_str(), self.open, self.close, self.high, self.low, self.trade_info)
    }
}

trait MetricModel {
    fn add(&self, klu: &mut CKLine_Unit);
}

impl MetricModel for CMACD {
    fn add(&self, klu: &mut CKLine_Unit) {
        klu.macd = Some(CMACD_item::new(klu.close));
    }
}

impl MetricModel for CTrendModel {
    fn add(&self, klu: &mut CKLine_Unit) {
        let trend_type = self.type;
        let trend_value = self.add(klu.close);
        klu.trend.entry(trend_type).or_insert(HashMap::new()).insert(self.T, trend_value);
    }
}

impl MetricModel for BollModel {
    fn add(&self, klu: &mut CKLine_Unit) {
        klu.boll = Some(self.add(klu.close));
    }
}

impl MetricModel for CDemarkEngine {
    fn add(&self, klu: &mut CKLine_Unit) {
        klu.demark = self.update(idx = klu.idx, close = klu.close, high = klu.high, low = klu.low);
    }
}

impl MetricModel for RSI {
    fn add(&self, klu: &mut CKLine_Unit) {
        klu.rsi = Some(self.add(klu.close));
    }
}

impl MetricModel for KDJ {
    fn add(&self, klu: &mut CKLine_Unit) {
        klu.kdj = Some(self.add(klu.high, klu.low, klu.close));
    }
}

#[derive(Debug, Clone)]
struct CKLine {
    idx: i32,
    kl_type: Option<String>,
    lst: Vec<CKLine_Unit>,
    dir: KLINE_DIR,
    fx: FX_TYPE,
    next: Option<Box<CKLine>>,
    pre: Option<Box<CKLine>>,
}

impl CKLine {
    fn new(kl_unit: CKLine_Unit, idx: i32, _dir: KLINE_DIR) -> Self {
        let mut kline = Self {
            idx,
            kl_type: kl_unit.kl_type.clone(),
            lst: vec![kl_unit.clone()],
            dir: _dir,
            fx: FX_TYPE::UNKNOWN,
            next: None,
            pre: None,
        };
        kl_unit.set_klc(Some(kline.clone()));
        kline
    }

    fn __str__(&self) -> String {
        let fx_token = match self.fx {
            FX_TYPE::TOP => "^",
            FX_TYPE::BOTTOM => "_",
            _ => "",
        };
        format!("{}th{}:{}~{}({}|{}) low={} high={}", self.idx, fx_token, self.time_begin(), self.time_end(), self.kl_type.as_ref().unwrap(), self.lst.len(), self.low(), self.high())
    }

    fn time_begin(&self) -> String {
        self.lst.first().unwrap().time.to_str().to_string()
    }

    fn time_end(&self) -> String {
        self.lst.last().unwrap().time.to_str().to_string()
    }

    fn low(&self) -> f64 {
        self.lst.iter().map(|klu| klu.low).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    fn high(&self) -> f64 {
        self.lst.iter().map(|klu| klu.high).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    fn get_klu_max_high(&self) -> f64 {
        self.lst.iter().map(|klu| klu.high).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    fn get_klu_min_low(&self) -> f64 {
        self.lst.iter().map(|klu| klu.low).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    fn has_gap_with_next(&self) -> bool {
        assert!(self.next.is_some());
        !has_overlap(self.get_klu_min_low(), self.get_klu_max_high(), self.next.as_ref().unwrap().get_klu_min_low(), self.next.as_ref().unwrap().get_klu_max_high(), true)
    }

    fn check_fx_valid(&self, item2: &CKLine, method: FX_CHECK_METHOD, for_virtual: bool) -> bool {
        assert!(self.next.is_some() && item2.pre.is_some());
        assert!(self.pre.is_some());
        assert!(item2.idx > self.idx);

        if self.fx == FX_TYPE::TOP {
            assert!(for_virtual || item2.fx == FX_TYPE::BOTTOM);
            if for_virtual && item2.dir != KLINE_DIR::DOWN {
                return false;
            }
            let (item2_high, self_low) = match method {
                FX_CHECK_METHOD::HALF => (item2.pre.as_ref().unwrap().high.max(item2.high), self.low.min(self.next.as_ref().unwrap().low)),
                FX_CHECK_METHOD::LOSS => (item2.high, self.low),
                FX_CHECK_METHOD::STRICT | FX_CHECK_METHOD::TOTALLY => {
                    if for_virtual {
                        (item2.pre.as_ref().unwrap().high.max(item2.high), self.pre.as_ref().unwrap().low.min(self.low).min(self.next.as_ref().unwrap().low))
                    } else {
                        assert!(item2.next.is_some());
                        (item2.pre.as_ref().unwrap().high.max(item2.high).max(item2.next.as_ref().unwrap().high), self.pre.as_ref().unwrap().low.min(self.low).min(self.next.as_ref().unwrap().low))
                    }
                }
            };
            if method == FX_CHECK_METHOD::TOTALLY {
                return self.low > item2_high;
            } else {
                return self.high > item2_high && item2.low < self_low;
            }
        } else if self.fx == FX_TYPE::BOTTOM {
            assert!(for_virtual || item2.fx == FX_TYPE::TOP);
            if for_virtual && item2.dir != KLINE_DIR::UP {
                return false;
            }
            let (item2_low, cur_high) = match method {
                FX_CHECK_METHOD::HALF => (item2.pre.as_ref().unwrap().low.min(item2.low), self.high.max(self.next.as_ref().unwrap().high)),
                FX_CHECK_METHOD::LOSS => (item2.low, self.high),
                FX_CHECK_METHOD::STRICT | FX_CHECK_METHOD::TOTALLY => {
                    if for_virtual {
                        (item2.pre.as_ref().unwrap().low.min(item2.low), self.pre.as_ref().unwrap().high.max(self.high).max(self.next.as_ref().unwrap().high))
                    } else {
                        assert!(item2.next.is_some());
                        (item2.pre.as_ref().unwrap().low.min(item2.low).min(item2.next.as_ref().unwrap().low), self.pre.as_ref().unwrap().high.max(self.high).max(self.next.as_ref().unwrap().high))
                    }
                }
            };
            if method == FX_CHECK_METHOD::TOTALLY {
                return self.high < item2_low;
            } else {
                return self.low < item2_low && item2.high > cur_high;
            }
        } else {
            panic!("only top/bottom fx can check_valid_top_button");
        }
    }
}

fn has_overlap(a_min: f64, a_max: f64, b_min: f64, b_max: f64, equal: bool) -> bool {
    if equal {
        a_min <= b_max && b_min <= a_max
    } else {
        a_min < b_max && b_min < a_max
    }
}

#[derive(Debug, Clone)]
struct CKLine_List {
    kl_type: String,
    config: CChanConfig,
    lst: Vec<CKLine>,
    bi_list: CBiList,
    seg_list: CSegListComm<CBi>,
    segseg_list: CSegListComm<CSeg<CBi>>,
    zs_list: CZSList,
    segzs_list: CZSList,
    bs_point_lst: CBSPointList<CBi, CBiList>,
    seg_bs_point_lst: CBSPointList<CSeg, CSegListComm>,
    metric_model_lst: Vec<Box<dyn MetricModel>>,
    step_calculation: bool,
}

impl CKLine_List {
    fn new(kl_type: String, conf: CChanConfig) -> Self {
        let bi_list = CBiList::new(conf.bi_conf.clone());
        let seg_list = get_seglist_instance(conf.seg_conf.clone(), SEG_TYPE::BI);
        let segseg_list = get_seglist_instance(conf.seg_conf.clone(), SEG_TYPE::SEG);
        let zs_list = CZSList::new(conf.zs_conf.clone());
        let segzs_list = CZSList::new(conf.zs_conf.clone());
        let bs_point_lst = CBSPointList::new(conf.bs_point_conf.clone());
        let seg_bs_point_lst = CBSPointList::new(conf.seg_bs_point_conf.clone());
        let metric_model_lst = conf.get_metric_model();
        let step_calculation = conf.trigger_step;

        Self {
            kl_type,
            config: conf,
            lst: Vec::new(),
            bi_list,
            seg_list,
            segseg_list,
            zs_list,
            segzs_list,
            bs_point_lst,
            seg_bs_point_lst,
            metric_model_lst,
            step_calculation,
        }
    }

    fn __deepcopy__(&self, memo: &mut HashMap<usize, CKLine_List>) -> CKLine_List {
        let mut new_obj = CKLine_List::new(self.kl_type.clone(), self.config.clone());
        memo.insert(self.lst.len(), new_obj.clone());

        for klc in &self.lst {
            let mut klus_new = Vec::new();
            for klu in &klc.lst {
                let new_klu = klu.clone();
                memo.insert(klu.lst.len(), new_klu.clone());
                if klu.pre.is_some() {
                    new_klu.set_pre_klu(memo[&klu.pre.unwrap().lst.len()].clone());
                }
                klus_new.push(new_klu);
            }

            let new_klc = CKLine::new(klus_new[0].clone(), klc.idx, klc.dir);
            new_klc.set_fx(klc.fx);
            new_klc.kl_type = klc.kl_type.clone();
            for (idx, klu) in klus_new.iter().enumerate() {
                klu.set_klc(Some(new_klc.clone()));
                if idx != 0 {
                    new_klc.add(klu.clone());
                }
            }
            memo.insert(klc.lst.len(), new_klc.clone());
            if new_obj.lst.len() > 0 {
                new_obj.lst.last_mut().unwrap().set_next(Some(Box::new(new_klc.clone())));
                new_klc.set_pre(Some(Box::new(new_obj.lst.last().unwrap().clone())));
            }
            new_obj.lst.push(new_klc);
        }

        new_obj.bi_list = self.bi_list.clone();
        new_obj.seg_list = self.seg_list.clone();
        new_obj.segseg_list = self.segseg_list.clone();
        new_obj.zs_list = self.zs_list.clone();
        new_obj.segzs_list = self.segzs_list.clone();
        new_obj.bs_point_lst = self.bs_point_lst.clone();
        new_obj.metric_model_lst = self.metric_model_lst.clone();
        new_obj.step_calculation = self.step_calculation;
        new_obj.seg_bs_point_lst = self.seg_bs_point_lst.clone();

        new_obj
    }

    fn __getitem__(&self, index: usize) -> &CKLine {
        &self.lst[index]
    }

    fn __len__(&self) -> usize {
        self.lst.len()
    }

    fn cal_seg_and_zs(&mut self) {
        if !self.step_calculation {
            self.bi_list.try_add_virtual_bi(self.lst.last().unwrap().clone());
        }
        cal_seg(&self.bi_list, &mut self.seg_list);
        self.zs_list.cal_bi_zs(&self.bi_list, &self.seg_list);
        update_zs_in_seg(&self.bi_list, &self.seg_list, &mut self.zs_list);

        cal_seg(&self.seg_list, &mut self.segseg_list);
        self.segzs_list.cal_bi_zs(&self.seg_list, &self.segseg_list);
        update_zs_in_seg(&self.seg_list, &self.segseg_list, &mut self.segzs_list);

        self.seg_bs_point_lst.cal(&self.seg_list, &self.segseg_list);
        self.bs_point_lst.cal(&self.bi_list, &self.seg_list);
    }

    fn need_cal_step_by_step(&self) -> bool {
        self.config.trigger_step
    }

    fn add_single_klu(&mut self, klu: CKLine_Unit) {
        klu.set_metric(self.metric_model_lst.clone());
        if self.lst.is_empty() {
            self.lst.push(CKLine::new(klu.clone(), 0, KLINE_DIR::UP));
        } else {
            let _dir = self.lst.last_mut().unwrap().try_add(klu.clone());
            if _dir != KLINE_DIR::COMBINE {
                self.lst.push(CKLine::new(klu.clone(), self.lst.len() as i32, _dir));
                if self.lst.len() >= 3 {
                    self.lst[self.lst.len() - 2].update_fx(self.lst[self.lst.len() - 3].clone(), self.lst.last().unwrap().clone());
                }
                if self.bi_list.update_bi(self.lst[self.lst.len() - 2].clone(), self.lst.last().unwrap().clone(), self.step_calculation) && self.step_calculation {
                    self.cal_seg_and_zs();
                }
            } else if self.step_calculation && self.bi_list.try_add_virtual_bi(self.lst.last().unwrap().clone(), true) {
                self.cal_seg_and_zs();
            }
        }
    }

    fn klu_iter(&self, klc_begin_idx: usize) -> impl Iterator<Item = &CKLine_Unit> {
        self.lst[klc_begin_idx..].iter().flat_map(|klc| klc.lst.iter())
    }
}

fn cal_seg(bi_list: &CBiList, seg_list: &mut CSegListComm<CBi>) {
    seg_list.update(bi_list);

    let mut sure_seg_cnt = 0;
    if seg_list.is_empty() {
        for bi in bi_list.iter() {
            bi.set_seg_idx(0);
        }
        return;
    }

    let mut begin_seg = seg_list.last().unwrap();
    for seg in seg_list.iter().rev() {
        if seg.is_sure {
            sure_seg_cnt += 1;
        } else {
            sure_seg_cnt = 0;
        }
        begin_seg = seg;
        if sure_seg_cnt > 2 {
            break;
        }
    }

    let mut cur_seg = seg_list.last().unwrap();
    for bi in bi_list.iter().rev() {
        if bi.seg_idx.is_some() && bi.idx < begin_seg.start_bi.idx {
            break;
        }
        if bi.idx > cur_seg.end_bi.idx {
            bi.set_seg_idx(cur_seg.idx + 1);
            continue;
        }
        if bi.idx < cur_seg.start_bi.idx {
            assert!(cur_seg.pre.is_some());
            cur_seg = cur_seg.pre.as_ref().unwrap();
        }
        bi.set_seg_idx(cur_seg.idx);
    }
}

fn update_zs_in_seg(bi_list: &CBiList, seg_list: &CSegListComm<CBi>, zs_list: &mut CZSList) {
    let mut sure_seg_cnt = 0;
    for seg in seg_list.iter().rev() {
        if seg.ele_inside_is_sure {
            break;
        }
        if seg.is_sure {
            sure_seg_cnt += 1;
        }
        seg.clear_zs_lst();
        for zs in zs_list.iter().rev() {
            if zs.end.idx < seg.start_bi.get_begin_klu().idx {
                break;
            }
            if zs.is_inside(seg) {
                seg.add_zs(zs);
            }
            assert!(zs.begin_bi.idx > 0);
            zs.set_bi_in(bi_list[zs.begin_bi.idx - 1].clone());
            if zs.end_bi.idx + 1 < bi_list.len() {
                zs.set_bi_out(bi_list[zs.end_bi.idx + 1].clone());
            }
            zs.set_bi_lst(bi_list[zs.begin_bi.idx..=zs.end_bi.idx].to_vec());
        }
        if sure_seg_cnt > 2 {
            if !seg.ele_inside_is_sure {
                seg.ele_inside_is_sure = true;
            }
        }
    }
}