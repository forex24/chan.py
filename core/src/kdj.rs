use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct KDJ_Item {
    k: f64,
    d: f64,
    j: f64,
}

impl KDJ_Item {
    pub fn new(k: f64, d: f64, j: f64) -> Self {
        Self { k, d, j }
    }
}

#[derive(Debug, Clone)]
pub struct KDJ {
    arr: Vec<HashMap<String, f64>>,
    period: usize,
    pre_kdj: KDJ_Item,
}

impl KDJ {
    pub fn new(period: usize) -> Self {
        Self {
            arr: Vec::new(),
            period,
            pre_kdj: KDJ_Item::new(50.0, 50.0, 50.0),
        }
    }

    pub fn add(&mut self, high: f64, low: f64, close: f64) -> KDJ_Item {
        self.arr.push(HashMap::from([
            ("high".to_string(), high),
            ("low".to_string(), low),
        ]));
        if self.arr.len() > self.period {
            self.arr.remove(0);
        }

        let hn = self
            .arr
            .iter()
            .map(|x| x["high"])
            .fold(f64::MIN, |a, b| a.max(b));
        let ln = self
            .arr
            .iter()
            .map(|x| x["low"])
            .fold(f64::MAX, |a, b| a.min(b));
        let cn = close;
        let rsv = if hn != ln {
            100.0 * (cn - ln) / (hn - ln)
        } else {
            0.0
        };

        let cur_k = 2.0 / 3.0 * self.pre_kdj.k + 1.0 / 3.0 * rsv;
        let cur_d = 2.0 / 3.0 * self.pre_kdj.d + 1.0 / 3.0 * cur_k;
        let cur_j = 3.0 * cur_k - 2.0 * cur_d;
        let cur_kdj = KDJ_Item::new(cur_k, cur_d, cur_j);
        self.pre_kdj = cur_kdj.clone();

        cur_kdj
    }
}
