use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, PartialEq)]
enum BI_DIR {
    UP,
    DOWN,
}

#[derive(Debug, Clone)]
struct C_KL {
    idx: i32,
    close: f64,
    high: f64,
    low: f64,
}

impl C_KL {
    fn v(&self, is_close: bool, _dir: BI_DIR) -> f64 {
        if is_close {
            self.close
        } else {
            match _dir {
                BI_DIR::UP => self.high,
                BI_DIR::DOWN => self.low,
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CDemarkIndex {
    data: Vec<HashMap<String, String>>,
}

impl CDemarkIndex {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn add(&mut self, _dir: BI_DIR, _type: &str, idx: i32, series: &CDemarkSetup) {
        let mut map = HashMap::new();
        map.insert("dir".to_string(), format!("{:?}", _dir));
        map.insert("type".to_string(), _type.to_string());
        map.insert("idx".to_string(), idx.to_string());
        map.insert("series".to_string(), format!("{:?}", series));
        self.data.push(map);
    }

    fn get_setup(&self) -> Vec<&HashMap<String, String>> {
        self.data.iter().filter(|info| info["type"] == "setup").collect()
    }

    fn get_countdown(&self) -> Vec<&HashMap<String, String>> {
        self.data.iter().filter(|info| info["type"] == "countdown").collect()
    }

    fn update(&mut self, demark_index: CDemarkIndex) {
        self.data.extend(demark_index.data);
    }
}

#[derive(Debug, Clone)]
struct CDemarkCountdown {
    dir: BI_DIR,
    kl_list: Vec<C_KL>,
    idx: i32,
    tdst_peak: f64,
    finish: bool,
}

impl CDemarkCountdown {
    fn new(_dir: BI_DIR, kl_list: Vec<C_KL>, tdst_peak: f64) -> Self {
        Self {
            dir: _dir,
            kl_list: kl_list.clone(),
            idx: 0,
            tdst_peak,
            finish: false,
        }
    }

    fn update(&mut self, kl: C_KL) -> bool {
        if self.finish {
            return false;
        }
        self.kl_list.push(kl.clone());
        if self.kl_list.len() <= CDemarkEngine::COUNTDOWN_BIAS {
            return false;
        }
        if self.idx == CDemarkEngine::MAX_COUNTDOWN {
            self.finish = true;
            return false;
        }
        if (self.dir == BI_DIR::DOWN && kl.high > self.tdst_peak) || (self.dir == BI_DIR::UP && kl.low < self.tdst_peak) {
            self.finish = true;
            return false;
        }
        if self.dir == BI_DIR::DOWN && self.kl_list.last().unwrap().close < self.kl_list[self.kl_list.len() - 1 - CDemarkEngine::COUNTDOWN_BIAS].v(CDemarkEngine::COUNTDOWN_CMP2CLOSE, self.dir) {
            self.idx += 1;
            return true;
        }
        if self.dir == BI_DIR::UP && self.kl_list.last().unwrap().close > self.kl_list[self.kl_list.len() - 1 - CDemarkEngine::COUNTDOWN_BIAS].v(CDemarkEngine::COUNTDOWN_CMP2CLOSE, self.dir) {
            self.idx += 1;
            return true;
        }
        false
    }
}

#[derive(Debug, Clone)]
struct CDemarkSetup {
    dir: BI_DIR,
    kl_list: Vec<C_KL>,
    pre_kl: C_KL,
    countdown: Option<CDemarkCountdown>,
    setup_finished: bool,
    idx: i32,
    tdst_peak: Option<f64>,
    last_demark_index: CDemarkIndex,
}

impl CDemarkSetup {
    fn new(_dir: BI_DIR, kl_list: Vec<C_KL>, pre_kl: C_KL) -> Self {
        assert_eq!(kl_list.len(), CDemarkEngine::SETUP_BIAS);
        Self {
            dir: _dir,
            kl_list: kl_list.clone(),
            pre_kl,
            countdown: None,
            setup_finished: false,
            idx: 0,
            tdst_peak: None,
            last_demark_index: CDemarkIndex::new(),
        }
    }

    fn update(&mut self, kl: C_KL) -> CDemarkIndex {
        self.last_demark_index = CDemarkIndex::new();
        if !self.setup_finished {
            self.kl_list.push(kl.clone());
            if self.dir == BI_DIR::DOWN {
                if self.kl_list.last().unwrap().close < self.kl_list[self.kl_list.len() - 1 - CDemarkEngine::SETUP_BIAS].v(CDemarkEngine::SETUP_CMP2CLOSE, self.dir) {
                    self.add_setup();
                } else {
                    self.setup_finished = true;
                }
            } else if self.kl_list.last().unwrap().close > self.kl_list[self.kl_list.len() - 1 - CDemarkEngine::SETUP_BIAS].v(CDemarkEngine::SETUP_CMP2CLOSE, self.dir) {
                self.add_setup();
            } else {
                self.setup_finished = true;
            }
        }
        if self.idx == CDemarkEngine::DEMARK_LEN && !self.setup_finished && self.countdown.is_none() {
            self.countdown = Some(CDemarkCountdown::new(self.dir, self.kl_list[..self.kl_list.len() - 1].to_vec(), self.cal_tdst_peak()));
        }
        if let Some(ref mut countdown) = self.countdown {
            if countdown.update(kl.clone()) {
                self.last_demark_index.add(self.dir, "countdown", countdown.idx, self);
            }
        }
        self.last_demark_index.clone()
    }

    fn add_setup(&mut self) {
        self.idx += 1;
        self.last_demark_index.add(self.dir, "setup", self.idx, self);
    }

    fn cal_tdst_peak(&mut self) -> f64 {
        assert_eq!(self.kl_list.len(), CDemarkEngine::SETUP_BIAS + CDemarkEngine::DEMARK_LEN);
        let arr = self.kl_list[CDemarkEngine::SETUP_BIAS..CDemarkEngine::SETUP_BIAS + CDemarkEngine::DEMARK_LEN].to_vec();
        assert_eq!(arr.len(), CDemarkEngine::DEMARK_LEN);
        let res = if self.dir == BI_DIR::DOWN {
            let max_high = arr.iter().map(|kl| kl.high).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            if CDemarkEngine::TIAOKONG_ST && arr[0].high < self.pre_kl.close {
                max_high.max(self.pre_kl.close)
            } else {
                max_high
            }
        } else {
            let min_low = arr.iter().map(|kl| kl.low).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            if CDemarkEngine::TIAOKONG_ST && arr[0].low > self.pre_kl.close {
                min_low.min(self.pre_kl.close)
            } else {
                min_low
            }
        };
        self.tdst_peak = Some(res);
        res
    }
}

#[derive(Debug, Clone)]
struct CDemarkEngine {
    demark_len: i32,
    setup_bias: i32,
    countdown_bias: i32,
    max_countdown: i32,
    tiaokong_st: bool,
    setup_cmp2close: bool,
    countdown_cmp2close: bool,
    kl_lst: Vec<C_KL>,
    series: Vec<CDemarkSetup>,
}

impl CDemarkEngine {
    const DEMARK_LEN: i32 = 9;
    const SETUP_BIAS: i32 = 4;
    const COUNTDOWN_BIAS: i32 = 2;
    const MAX_COUNTDOWN: i32 = 13;
    const TIAOKONG_ST: bool = true;
    const SETUP_CMP2CLOSE: bool = true;
    const COUNTDOWN_CMP2CLOSE: bool = true;

    fn new(
        demark_len: i32,
        setup_bias: i32,
        countdown_bias: i32,
        max_countdown: i32,
        tiaokong_st: bool,
        setup_cmp2close: bool,
        countdown_cmp2close: bool,
    ) -> Self {
        Self {
            demark_len,
            setup_bias,
            countdown_bias,
            max_countdown,
            tiaokong_st,
            setup_cmp2close,
            countdown_cmp2close,
            kl_lst: Vec::new(),
            series: Vec::new(),
        }
    }

    fn update(&mut self, idx: i32, close: f64, high: f64, low: f64) -> CDemarkIndex {
        self.kl_lst.push(C_KL { idx, close, high, low });
        if self.kl_lst.len() <= (self.setup_bias + 1) as usize {
            return CDemarkIndex::new();
        }

        if self.kl_lst.last().unwrap().close < self.kl_lst[self.kl_lst.len() - 1 - self.setup_bias as usize].close {
            if !self.series.iter().any(|series| series.dir == BI_DIR::DOWN && !series.setup_finished) {
                self.series.push(CDemarkSetup::new(BI_DIR::DOWN, self.kl_lst[self.kl_lst.len() - self.setup_bias as usize - 1..self.kl_lst.len() - 1].to_vec(), self.kl_lst[self.kl_lst.len() - self.setup_bias as usize - 2].clone()));
            }
            for series in &mut self.series {
                if series.dir == BI_DIR::UP && series.countdown.is_none() && !series.setup_finished {
                    series.setup_finished = true;
                }
            }
        } else if self.kl_lst.last().unwrap().close > self.kl_lst[self.kl_lst.len() - 1 - self.setup_bias as usize].close {
            if !self.series.iter().any(|series| series.dir == BI_DIR::UP && !series.setup_finished) {
                self.series.push(CDemarkSetup::new(BI_DIR::UP, self.kl_lst[self.kl_lst.len() - self.setup_bias as usize - 1..self.kl_lst.len() - 1].to_vec(), self.kl_lst[self.kl_lst.len() - self.setup_bias as usize - 2].clone()));
            }
            for series in &mut self.series {
                if series.dir == BI_DIR::DOWN && series.countdown.is_none() && !series.setup_finished {
                    series.setup_finished = true;
                }
            }
        }

        self.clear();
        self.clean_series_from_setup_finish();

        let result = self.cal_result();
        self.clear();
        result
    }

    fn cal_result(&self) -> CDemarkIndex {
        let mut demark_index = CDemarkIndex::new();
        for series in &self.series {
            demark_index.update(series.last_demark_index.clone());
        }
        demark_index
    }

    fn clear(&mut self) {
        self.series.retain(|series| !(series.setup_finished && series.countdown.is_none()));
        self.series.retain(|series| !(series.countdown.is_some() && series.countdown.as_ref().unwrap().finish));
    }

    fn clean_series_from_setup_finish(&mut self) {
        let mut finished_setup: Option<usize> = None;
        for series in &mut self.series {
            let demark_idx = series.update(self.kl_lst.last().unwrap().clone());
            for setup_idx in demark_idx.get_setup() {
                if setup_idx["idx"].parse::<i32>().unwrap() == CDemarkEngine::DEMARK_LEN {
                    assert!(finished_setup.is_none());
                    finished_setup = Some(series as *const CDemarkSetup as usize);
                }
            }
        }
        if let Some(finished_setup) = finished_setup {
            self.series.retain(|series| series as *const CDemarkSetup as usize == finished_setup);
        }
    }
}