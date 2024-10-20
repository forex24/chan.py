use std::collections::HashMap;
use std::fmt;

use crate::{
    cenum::{FX_CHECK_METHOD, FX_TYPE, KLINE_DIR},
    klu::CKLine_Unit,
};
/*
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
*/
#[derive(Debug, Clone)]
pub struct CKLine {
    pub idx: i32,
    pub kl_type: Option<String>,
    pub lst: Vec<CKLine_Unit>,
    pub dir: KLINE_DIR,
    pub fx: FX_TYPE,
    pub next: Option<Box<CKLine>>,
    pub pre: Option<Box<CKLine>>,
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
        format!(
            "{}th{}:{}~{}({}|{}) low={} high={}",
            self.idx,
            fx_token,
            self.time_begin(),
            self.time_end(),
            self.kl_type.as_ref().unwrap(),
            self.lst.len(),
            self.low(),
            self.high()
        )
    }

    fn time_begin(&self) -> String {
        self.lst.first().unwrap().time.to_str().to_string()
    }

    fn time_end(&self) -> String {
        self.lst.last().unwrap().time.to_str().to_string()
    }

    fn low(&self) -> f64 {
        self.lst
            .iter()
            .map(|klu| klu.low)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn high(&self) -> f64 {
        self.lst
            .iter()
            .map(|klu| klu.high)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn get_klu_max_high(&self) -> f64 {
        self.lst
            .iter()
            .map(|klu| klu.high)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn get_klu_min_low(&self) -> f64 {
        self.lst
            .iter()
            .map(|klu| klu.low)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn has_gap_with_next(&self) -> bool {
        assert!(self.next.is_some());
        !has_overlap(
            self.get_klu_min_low(),
            self.get_klu_max_high(),
            self.next.as_ref().unwrap().get_klu_min_low(),
            self.next.as_ref().unwrap().get_klu_max_high(),
            true,
        )
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
                FX_CHECK_METHOD::HALF => (
                    item2.pre.as_ref().unwrap().high.max(item2.high),
                    self.low.min(self.next.as_ref().unwrap().low),
                ),
                FX_CHECK_METHOD::LOSS => (item2.high, self.low),
                FX_CHECK_METHOD::STRICT | FX_CHECK_METHOD::TOTALLY => {
                    if for_virtual {
                        (
                            item2.pre.as_ref().unwrap().high.max(item2.high),
                            self.pre
                                .as_ref()
                                .unwrap()
                                .low
                                .min(self.low)
                                .min(self.next.as_ref().unwrap().low),
                        )
                    } else {
                        assert!(item2.next.is_some());
                        (
                            item2
                                .pre
                                .as_ref()
                                .unwrap()
                                .high
                                .max(item2.high)
                                .max(item2.next.as_ref().unwrap().high),
                            self.pre
                                .as_ref()
                                .unwrap()
                                .low
                                .min(self.low)
                                .min(self.next.as_ref().unwrap().low),
                        )
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
                FX_CHECK_METHOD::HALF => (
                    item2.pre.as_ref().unwrap().low.min(item2.low),
                    self.high.max(self.next.as_ref().unwrap().high),
                ),
                FX_CHECK_METHOD::LOSS => (item2.low, self.high),
                FX_CHECK_METHOD::STRICT | FX_CHECK_METHOD::TOTALLY => {
                    if for_virtual {
                        (
                            item2.pre.as_ref().unwrap().low.min(item2.low),
                            self.pre
                                .as_ref()
                                .unwrap()
                                .high
                                .max(self.high)
                                .max(self.next.as_ref().unwrap().high),
                        )
                    } else {
                        assert!(item2.next.is_some());
                        (
                            item2
                                .pre
                                .as_ref()
                                .unwrap()
                                .low
                                .min(item2.low)
                                .min(item2.next.as_ref().unwrap().low),
                            self.pre
                                .as_ref()
                                .unwrap()
                                .high
                                .max(self.high)
                                .max(self.next.as_ref().unwrap().high),
                        )
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
