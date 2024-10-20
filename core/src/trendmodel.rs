use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TREND_TYPE {
    MEAN,
    MAX,
    MIN,
}
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
*/
#[derive(Debug, Clone)]
pub struct CTrendModel {
    pub t: usize,
    pub arr: Vec<f64>,
    pub trend_type: TREND_TYPE,
}

impl CTrendModel {
    pub fn new(trend_type: TREND_TYPE, t: usize) -> Self {
        Self {
            t,
            arr: Vec::new(),
            trend_type,
        }
    }

    pub fn add(&mut self, value: f64) -> f64 {
        self.arr.push(value);
        if self.arr.len() > self.t {
            self.arr.drain(0..self.arr.len() - self.t);
        }
        match self.trend_type {
            TREND_TYPE::MEAN => self.arr.iter().sum::<f64>() / self.arr.len() as f64,
            TREND_TYPE::MAX => *self
                .arr
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            TREND_TYPE::MIN => *self
                .arr
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        }
    }
}
