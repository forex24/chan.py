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
enum DATA_FIELD {
    FIELD_TIME,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_HIGH,
    FIELD_LOW,
}

#[derive(Debug, Clone, PartialEq)]
enum TRADE_INFO_LST {
    // Define your trade info fields here
}

#[derive(Debug, Clone, PartialEq)]
enum TREND_TYPE {
    // Define your trend types here
}

#[derive(Debug, Clone)]
struct CTime {
    time: String,
}

impl CTime {
    fn to_str(&self) -> &str {
        &self.time
    }
}

#[derive(Debug, Clone)]
struct CTradeInfo {
    metric: HashMap<String, f64>,
}

impl CTradeInfo {
    fn new(kl_dict: HashMap<DATA_FIELD, f64>) -> Self {
        let mut metric = HashMap::new();
        for (key, value) in kl_dict {
            metric.insert(format!("{:?}", key), value);
        }
        Self { metric }
    }
}

#[derive(Debug, Clone)]
struct CDemarkIndex {
    // Define your demark index fields here
}

#[derive(Debug, Clone)]
struct CMACD_item {
    // Define your MACD item fields here
}

#[derive(Debug, Clone)]
struct BOLL_Metric {
    // Define your BOLL metric fields here
}

#[derive(Debug, Clone)]
struct CTrendModel {
    // Define your trend model fields here
}

#[derive(Debug, Clone)]
struct BollModel {
    // Define your BOLL model fields here
}

#[derive(Debug, Clone)]
struct CDemarkEngine {
    // Define your demark engine fields here
}

#[derive(Debug, Clone)]
struct RSI {
    // Define your RSI fields here
}

#[derive(Debug, Clone)]
struct KDJ {
    // Define your KDJ fields here
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