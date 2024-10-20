use std::f64;
fn _truncate(x: f64) -> f64 {
    if x == 0.0 {
        1e-7
    } else {
        x
    }
}

#[derive(Debug, Clone)]
pub struct BOLL_Metric {
    pub theta: f64,
    pub up: f64,
    pub down: f64,
    pub mid: f64,
}

impl BOLL_Metric {
    pub fn new(ma: f64, theta: f64) -> Self {
        let theta = _truncate(theta);
        Self {
            theta,
            up: ma + 2.0 * theta,
            down: _truncate(ma - 2.0 * theta),
            mid: ma,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BollModel {
    pub n: usize,
    pub arr: Vec<f64>,
}

impl BollModel {
    pub fn new(n: usize) -> Self {
        assert!(n > 1);
        Self { n, arr: Vec::new() }
    }

    pub fn add(&mut self, value: f64) -> BOLL_Metric {
        self.arr.push(value);
        if self.arr.len() > self.n {
            self.arr.drain(0..self.arr.len() - self.n);
        }
        let ma = self.arr.iter().sum::<f64>() / self.arr.len() as f64;
        let theta = (self.arr.iter().map(|&x| (x - ma).powi(2)).sum::<f64>()
            / self.arr.len() as f64)
            .sqrt();
        BOLL_Metric::new(ma, theta)
    }
}
