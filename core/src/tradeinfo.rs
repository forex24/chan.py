use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct CTradeInfo {
    metric: HashMap<String, Option<f64>>,
}

impl CTradeInfo {
    pub fn new(info: HashMap<String, f64>) -> Self {
        let mut metric = HashMap::new();
        for metric_name in TRADE_INFO_LST {
            metric.insert(
                metric_name.to_string(),
                info.get(&metric_name.to_string()).cloned(),
            );
        }
        Self { metric }
    }
}

impl fmt::Display for CTradeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        for (metric_name, value) in &self.metric {
            if let Some(val) = value {
                parts.push(format!("{}:{}", metric_name, val));
            }
        }
        write!(f, "{}", parts.join(" "))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TRADE_INFO_LST {
    // Define your trade info fields here
}

impl TRADE_INFO_LST {
    pub fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}
