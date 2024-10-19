use std::collections::HashMap;

#[derive(Debug, Clone)]
struct CBi {
    begin_klc: CKLineUnit,
    end_klc: CKLineUnit,
    high: f64,
    low: f64,
    dir: i32,
    idx: i64,
    next: Option<Box<CBi>>,
    is_sure: bool,
    bsp: Option<CBS_Point<CBi>>,
}

impl CBi {
    fn _high(&self) -> f64 {
        self.high
    }

    fn _low(&self) -> f64 {
        self.low
    }

    fn is_down(&self) -> bool {
        self.dir == 1
    }

    fn is_up(&self) -> bool {
        self.dir == -1
    }

    fn get_end_val(&self) -> f64 {
        self.end_klc.high
    }

    fn get_begin_val(&self) -> f64 {
        self.begin_klc.high
    }

    fn get_end_klu(&self) -> CKLineUnit {
        self.end_klc.clone()
    }

    fn get_begin_klu(&self) -> CKLineUnit {
        self.begin_klc.clone()
    }

    fn amp(&self) -> f64 {
        (self.get_end_val() - self.get_begin_val()).abs()
    }
}

#[derive(Debug, Clone)]
struct CKLineUnit {
    time: i64,
    high: f64,
    low: f64,
}

#[derive(Debug, Clone)]
struct CSeg<T: Clone> {
    idx: i64,
    start_bi: T,
    end_bi: T,
    is_sure: bool,
    dir: i32,
    zs_lst: Vec<CZS<T>>,
    eigen_fx: Option<CEigenFX>,
    seg_idx: Option<i64>,
    parent_seg: Option<Box<CSeg<T>>>,
    pre: Option<Box<CSeg<T>>>,
    next: Option<Box<CSeg<T>>>,
    bsp: Option<CBS_Point<CSeg<T>>>,
    bi_list: Vec<T>,
    reason: String,
    support_trend_line: Option<CTrendLine>,
    resistance_trend_line: Option<CTrendLine>,
    ele_inside_is_sure: bool,
}

impl<T: Clone + PartialEq> CSeg<T> {
    fn new(
        idx: i64,
        start_bi: T,
        end_bi: T,
        is_sure: bool,
        seg_dir: Option<i32>,
        reason: &str,
    ) -> Result<Self, CChanException> {
        assert!(
            start_bi.idx == 0 || start_bi.dir == end_bi.dir || !is_sure,
            format!(
                "{} {} {} {}",
                start_bi.idx, end_bi.idx, start_bi.dir, end_bi.dir
            )
        );
        let dir = seg_dir.unwrap_or(end_bi.dir);
        let mut seg = Self {
            idx,
            start_bi: start_bi.clone(),
            end_bi: end_bi.clone(),
            is_sure,
            dir,
            zs_lst: Vec::new(),
            eigen_fx: None,
            seg_idx: None,
            parent_seg: None,
            pre: None,
            next: None,
            bsp: None,
            bi_list: Vec::new(),
            reason: reason.to_string(),
            support_trend_line: None,
            resistance_trend_line: None,
            ele_inside_is_sure: false,
        };
        if end_bi.idx - start_bi.idx < 2 {
            seg.is_sure = false;
        }
        seg.check()?;
        Ok(seg)
    }

    // 其他方法省略，因为它们不直接用于 CBS_Point
}

#[derive(Debug, Clone)]
enum BSP_TYPE {
    Type1,
    Type2,
    Type3,
}

impl BSP_TYPE {
    fn value(&self) -> &'static str {
        match self {
            BSP_TYPE::Type1 => "Type1",
            BSP_TYPE::Type2 => "Type2",
            BSP_TYPE::Type3 => "Type3",
        }
    }
}

#[derive(Debug, Clone)]
struct CFeatures {
    features: HashMap<String, f64>,
}

impl CFeatures {
    fn new(feature_dict: Option<HashMap<String, f64>>) -> Self {
        Self {
            features: feature_dict.unwrap_or_default(),
        }
    }

    fn add_feat(&mut self, inp1: impl Into<String>, inp2: Option<f64>) {
        let key = inp1.into();
        if let Some(value) = inp2 {
            self.features.insert(key, value);
        }
    }
}

#[derive(Debug, Clone)]
struct CBS_Point<T: Clone> {
    bi: T,
    klu: CKLineUnit,
    is_buy: bool,
    type_: Vec<BSP_TYPE>,
    relate_bsp1: Option<Box<CBS_Point<T>>>,
    features: CFeatures,
    is_segbsp: bool,
}

impl<T: Clone> CBS_Point<T> {
    fn new(
        bi: T,
        is_buy: bool,
        bs_type: BSP_TYPE,
        relate_bsp1: Option<Box<CBS_Point<T>>>,
        feature_dict: Option<HashMap<String, f64>>,
    ) -> Self {
        let klu = bi.get_end_klu();
        let mut bsp = Self {
            bi: bi.clone(),
            klu,
            is_buy,
            type_: vec![bs_type],
            relate_bsp1,
            features: CFeatures::new(feature_dict),
            is_segbsp: false,
        };
        bsp.bi.bsp = Some(bsp.clone());
        bsp.init_common_feature();
        bsp
    }

    fn add_type(&mut self, bs_type: BSP_TYPE) {
        self.type_.push(bs_type);
    }

    fn type2str(&self) -> String {
        self.type_
            .iter()
            .map(|x| x.value())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn add_another_bsp_prop(&mut self, bs_type: BSP_TYPE, relate_bsp1: Option<Box<CBS_Point<T>>>) {
        self.add_type(bs_type);
        if self.relate_bsp1.is_none() {
            self.relate_bsp1 = relate_bsp1;
        } else if let Some(relate_bsp1) = relate_bsp1 {
            assert_eq!(
                self.relate_bsp1.as_ref().unwrap().klu.idx,
                relate_bsp1.klu.idx
            );
        }
    }

    fn add_feat(&mut self, inp1: impl Into<String>, inp2: Option<f64>) {
        self.features.add_feat(inp1, inp2);
    }

    fn init_common_feature(&mut self) {
        self.add_feat("bsp_bi_amp", Some(self.bi.amp()));
    }
}
