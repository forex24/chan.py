#[derive(Debug, Clone, PartialEq)]
pub enum KLINE_DIR {
    UP,
    DOWN,
    COMBINE,
    INCLUDED,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FX_CHECK_METHOD {
    STRICT,
    LOSS,
    HALF,
    TOTALLY,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BI_DIR {
    UP,
    DOWN,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BI_TYPE {
    STRICT,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FX_TYPE {
    BOTTOM,
    TOP,
    UNKNOWN,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MACD_ALGO {
    AREA,
    PEAK,
    FULL_AREA,
    DIFF,
    SLOPE,
    AMP,
    AMOUNT,
    VOLUMN,
    VOLUMN_AVG,
    AMOUNT_AVG,
    TURNRATE_AVG,
    RSI,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DATA_FIELD {
    FIELD_TURNOVER,
    FIELD_VOLUME,
    FIELD_TURNRATE,
}

#[derive(Debug, Clone)]
pub enum BSP_TYPE {
    T1,
    T1P,
    T2,
    T2S,
    T3A,
    T3B,
}

impl BSP_TYPE {
    pub fn value(&self) -> &'static str {
        match self {
            BSP_TYPE::T1 => "T1",
            BSP_TYPE::T1P => "T1P",
            BSP_TYPE::T2 => "T2",
            BSP_TYPE::T2S => "T2S",
            BSP_TYPE::T3A => "T3A",
            BSP_TYPE::T3B => "T3B",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CFeatures {
    features: HashMap<String, f64>,
}

impl CFeatures {
    pub fn new(feature_dict: Option<HashMap<String, f64>>) -> Self {
        Self {
            features: feature_dict.unwrap_or_default(),
        }
    }

    pub fn add_feat(&mut self, inp1: impl Into<String>, inp2: Option<f64>) {
        let key = inp1.into();
        if let Some(value) = inp2 {
            self.features.insert(key, value);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CChanException {
    pub message: String,
    pub err_code: i32,
}

impl CChanException {
    pub fn new(message: String, err_code: i32) -> Self {
        Self { message, err_code }
    }
}

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct CTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub auto: bool,
    pub ts: u64,
}

impl CTime {
    pub fn new(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
        auto: bool,
    ) -> Self {
        let mut time = Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            auto,
            ts: 0,
        };
        time.set_timestamp();
        time
    }

    pub fn to_str(&self) -> String {
        if self.hour == 0 && self.minute == 0 {
            format!("{:04}/{:02}/{:02}", self.year, self.month, self.day)
        } else {
            format!(
                "{:04}/{:02}/{:02} {:02}:{:02}",
                self.year, self.month, self.day, self.hour, self.minute
            )
        }
    }

    pub fn to_date_str(&self, splt: &str) -> String {
        format!(
            "{:04}{}{:02}{}{:02}",
            self.year, splt, self.month, splt, self.day
        )
    }

    pub fn to_date(&self) -> Self {
        Self::new(self.year, self.month, self.day, 0, 0, 0, false)
    }

    fn set_timestamp(&mut self) {
        let date = if self.hour == 0 && self.minute == 0 && self.auto {
            SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(self.ts)
        } else {
            SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(self.ts)
        };
        self.ts = date
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
    }
}

impl PartialEq for CTime {
    fn eq(&self, other: &Self) -> bool {
        self.ts == other.ts
    }
}

impl PartialOrd for CTime {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.ts.partial_cmp(&other.ts)
    }
}

impl std::fmt::Display for CTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_str() {
        let time = CTime::new(2023, 10, 5, 12, 30, 0, true);
        assert_eq!(time.to_str(), "2023/10/05 12:30");
    }

    #[test]
    fn test_to_date_str() {
        let time = CTime::new(2023, 10, 5, 12, 30, 0, true);
        assert_eq!(time.to_date_str("-"), "2023-10-05");
    }

    #[test]
    fn test_to_date() {
        let time = CTime::new(2023, 10, 5, 12, 30, 0, true);
        let date = time.to_date();
        assert_eq!(date.to_str(), "2023/10/05");
    }

    #[test]
    fn test_comparison() {
        let time1 = CTime::new(2023, 10, 5, 12, 30, 0, true);
        let time2 = CTime::new(2023, 10, 5, 13, 30, 0, true);
        assert!(time1 < time2);
        assert!(time2 > time1);
        assert!(time1 <= time2);
        assert!(time2 >= time1);
    }
}
