#[derive(Debug, Clone)]
struct RSI {
    close_arr: Vec<f64>,
    period: usize,
    diff: Vec<f64>,
    up: Vec<f64>,
    down: Vec<f64>,
}

impl RSI {
    fn new(period: usize) -> Self {
        Self {
            close_arr: Vec::new(),
            period,
            diff: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
        }
    }

    fn add(&mut self, close: f64) -> f64 {
        self.close_arr.push(close);
        if self.close_arr.len() == 1 {
            return 50.0;
        }
        self.diff
            .push(self.close_arr.last().unwrap() - self.close_arr[self.close_arr.len() - 2]);
        if self.diff.len() < self.period {
            let up_sum: f64 = self.diff.iter().filter(|&x| *x > 0.0).sum();
            let down_sum: f64 = self.diff.iter().filter(|&x| *x < 0.0).map(|x| -x).sum();
            self.up.push(up_sum / self.period as f64);
            self.down.push(down_sum / self.period as f64);
        } else {
            let upval = if self.diff.last().unwrap() > &0.0 {
                *self.diff.last().unwrap()
            } else {
                0.0
            };
            let downval = if self.diff.last().unwrap() < &0.0 {
                -self.diff.last().unwrap()
            } else {
                0.0
            };
            self.up.push(
                (self.up.last().unwrap() * (self.period - 1) as f64 + upval) / self.period as f64,
            );
            self.down.push(
                (self.down.last().unwrap() * (self.period - 1) as f64 + downval)
                    / self.period as f64,
            );
        }
        let rs = if self.down.last().unwrap() != &0.0 {
            self.up.last().unwrap() / self.down.last().unwrap()
        } else {
            0.0
        };
        let rsi = 100.0 - 100.0 / (1.0 + rs);
        rsi
    }
}
