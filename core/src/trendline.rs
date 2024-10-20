use std::cmp::Ordering;
use std::f64;

use crate::bi::CBi;
use crate::cenum::BI_DIR;
/*
#[derive(Debug, Clone, PartialEq)]
enum BI_DIR {
    UP,
    DOWN,
}
*/
#[derive(Debug, Clone, PartialEq)]
pub enum TREND_LINE_SIDE {
    INSIDE,
    OUTSIDE,
}

#[derive(Debug, Clone)]
pub struct Point {
    x: i32,
    y: f64,
}

impl Point {
    pub fn cal_slope(&self, p: &Point) -> f64 {
        if self.x != p.x {
            (self.y - p.y) / (self.x - p.x)
        } else {
            f64::INFINITY
        }
    }
}

#[derive(Debug, Clone)]
pub struct Line {
    p: Point,
    slope: f64,
}

impl Line {
    pub fn cal_dis(&self, p: &Point) -> f64 {
        (self.slope * p.x - p.y + self.p.y - self.slope * self.p.x).abs()
            / (self.slope.powi(2) + 1.0).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct CTrendLine {
    line: Option<Line>,
    side: TREND_LINE_SIDE,
}

impl CTrendLine {
    pub fn new(lst: Vec<CBi>, side: TREND_LINE_SIDE) -> Self {
        let mut trend_line = Self { line: None, side };
        trend_line.cal(lst);
        trend_line
    }

    pub fn cal(&mut self, lst: Vec<CBi>) {
        let mut bench = f64::INFINITY;
        let all_p = if self.side == TREND_LINE_SIDE::INSIDE {
            lst.iter()
                .rev()
                .step_by(2)
                .map(|bi| Point {
                    x: bi.get_begin_klu().idx,
                    y: bi.get_begin_val(),
                })
                .collect()
        } else {
            lst.iter()
                .rev()
                .step_by(2)
                .map(|bi| Point {
                    x: bi.get_end_klu().idx,
                    y: bi.get_end_val(),
                })
                .collect()
        };
        let mut c_p = all_p.clone();
        while !c_p.is_empty() {
            let (line, idx) = cal_tl(&c_p, lst.last().unwrap().dir, self.side);
            let dis = all_p.iter().map(|p| line.cal_dis(p)).sum::<f64>();
            if dis < bench {
                bench = dis;
                self.line = Some(line);
            }
            c_p = c_p[idx..].to_vec();
            if c_p.len() == 1 {
                break;
            }
        }
    }
}

pub fn init_peak_slope(_dir: BI_DIR, side: TREND_LINE_SIDE) -> f64 {
    match side {
        TREND_LINE_SIDE::INSIDE => 0.0,
        TREND_LINE_SIDE::OUTSIDE => match _dir {
            BI_DIR::UP => f64::INFINITY,
            BI_DIR::DOWN => f64::NEG_INFINITY,
        },
    }
}

pub fn cal_tl(c_p: &[Point], _dir: BI_DIR, side: TREND_LINE_SIDE) -> (Line, usize) {
    let p = c_p[0].clone();
    let mut peak_slope = init_peak_slope(_dir, side);
    let mut idx = 1;
    for (point_idx, p2) in c_p[1..].iter().enumerate() {
        let slope = p.cal_slope(p2);
        if (_dir == BI_DIR::UP && slope < 0.0) || (_dir == BI_DIR::DOWN && slope > 0.0) {
            continue;
        }
        match side {
            TREND_LINE_SIDE::INSIDE => {
                if (_dir == BI_DIR::UP && slope > peak_slope)
                    || (_dir == BI_DIR::DOWN && slope < peak_slope)
                {
                    peak_slope = slope;
                    idx = point_idx + 1;
                }
            }
            TREND_LINE_SIDE::OUTSIDE => {
                if (_dir == BI_DIR::UP && slope < peak_slope)
                    || (_dir == BI_DIR::DOWN && slope > peak_slope)
                {
                    peak_slope = slope;
                    idx = point_idx + 1;
                }
            }
        }
    }
    (
        Line {
            p,
            slope: peak_slope,
        },
        idx,
    )
}
/*
#[derive(Debug, Clone)]
struct BI {
    dir: BI_DIR,
    begin_klu: Point,
    end_klu: Point,
}

impl BI {
    pub fn get_begin_klu(&self) -> &Point {
        &self.begin_klu
    }

    pub fn get_end_klu(&self) -> &Point {
        &self.end_klu
    }

    pub fn get_begin_val(&self) -> f64 {
        self.begin_klu.y
    }

    pub fn get_end_val(&self) -> f64 {
        self.end_klu.y
    }
}
*/
