#[derive(Debug, Clone)]
struct CMACD_item {
    fast_ema: f64,
    slow_ema: f64,
    dif: f64,
    dea: f64,
    macd: f64,
}

impl CMACD_item {
    fn new(fast_ema: f64, slow_ema: f64, dif: f64, dea: f64) -> Self {
        let macd = 2.0 * (dif - dea);
        Self {
            fast_ema,
            slow_ema,
            dif,
            dea,
            macd,
        }
    }
}

#[derive(Debug, Clone)]
struct CMACD {
    macd_info: Vec<CMACD_item>,
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
}

impl CMACD {
    fn new(fastperiod: usize, slowperiod: usize, signalperiod: usize) -> Self {
        Self {
            macd_info: Vec::new(),
            fastperiod,
            slowperiod,
            signalperiod,
        }
    }

    fn add(&mut self, value: f64) -> CMACD_item {
        if self.macd_info.is_empty() {
            self.macd_info.push(CMACD_item::new(value, value, 0.0, 0.0));
        } else {
            let last_item = &self.macd_info.last().unwrap();
            let fast_ema = (2.0 * value + (self.fastperiod - 1) as f64 * last_item.fast_ema)
                / (self.fastperiod + 1) as f64;
            let slow_ema = (2.0 * value + (self.slowperiod - 1) as f64 * last_item.slow_ema)
                / (self.slowperiod + 1) as f64;
            let dif = fast_ema - slow_ema;
            let dea = (2.0 * dif + (self.signalperiod - 1) as f64 * last_item.dea)
                / (self.signalperiod + 1) as f64;
            self.macd_info
                .push(CMACD_item::new(fast_ema, slow_ema, dif, dea));
        }
        self.macd_info.last().unwrap().clone()
    }
}
