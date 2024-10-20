use std::collections::HashMap;

use crate::{bsp::CBS_Point, cenum::BSP_TYPE, seg::CSeg};

/*#[derive(Debug, Clone)]
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
    pub fn _high(&self) -> f64 {
        self.high
    }

    pub fn _low(&self) -> f64 {
        self.low
    }

    pub fn is_down(&self) -> bool {
        self.dir == 1
    }

    pub fn is_up(&self) -> bool {
        self.dir == -1
    }

    pub fn get_end_val(&self) -> f64 {
        self.end_klc.high
    }

    pub fn get_begin_val(&self) -> f64 {
        self.begin_klc.high
    }

    pub fn get_end_klu(&self) -> CKLineUnit {
        self.end_klc.clone()
    }

    pub fn get_begin_klu(&self) -> CKLineUnit {
        self.begin_klc.clone()
    }

    pub fn amp(&self) -> f64 {
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
    pub fn new(
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

    // 其他方法省略，因为它们不直接用于 CBSPointList
}

#[derive(Debug, Clone)]
enum BSP_TYPE {
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
    pub fn new(
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

    pub fn add_type(&mut self, bs_type: BSP_TYPE) {
        self.type_.push(bs_type);
    }

    pub fn type2str(&self) -> String {
        self.type_
            .iter()
            .map(|x| x.value())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn add_another_bsp_prop(
        &mut self,
        bs_type: BSP_TYPE,
        relate_bsp1: Option<Box<CBS_Point<T>>>,
    ) {
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

    pub fn add_feat(&mut self, inp1: impl Into<String>, inp2: Option<f64>) {
        self.features.add_feat(inp1, inp2);
    }

    pub fn init_common_feature(&mut self) {
        self.add_feat("bsp_bi_amp", Some(self.bi.amp()));
    }
}
*/

#[derive(Debug, Clone)]
pub struct CBSPointConfig {
    pub target_types: Vec<BSP_TYPE>,
    pub min_zs_cnt: i32,
    pub bsp1_only_multibi_zs: bool,
    pub bs1_peak: bool,
    pub macd_algo: i32,
    pub divergence_rate: f64,
    pub max_bs2_rate: f64,
    pub bsp2_follow_1: bool,
    pub bsp2s_follow_2: bool,
    pub max_bsp2s_lv: Option<i32>,
    pub strict_bsp3: bool,
    pub bsp3_peak: bool,
    pub bsp3_follow_1: bool,
}

impl CBSPointConfig {
    pub fn new() -> Self {
        Self {
            target_types: vec![
                BSP_TYPE::T1,
                BSP_TYPE::T1P,
                BSP_TYPE::T2,
                BSP_TYPE::T2S,
                BSP_TYPE::T3A,
                BSP_TYPE::T3B,
            ],
            min_zs_cnt: 0,
            bsp1_only_multibi_zs: false,
            bs1_peak: false,
            macd_algo: 0,
            divergence_rate: 0.0,
            max_bs2_rate: 0.0,
            bsp2_follow_1: false,
            bsp2s_follow_2: false,
            max_bsp2s_lv: None,
            strict_bsp3: false,
            bsp3_peak: false,
            bsp3_follow_1: false,
        }
    }

    pub fn get_bs_config(&self, is_down: bool) -> &CBSPointConfig {
        self
    }
}

#[derive(Debug, Clone)]
pub struct CBSPointList<T: Clone, L: Clone> {
    pub lst: Vec<CBS_Point<T>>,
    pub bsp_dict: HashMap<i64, CBS_Point<T>>,
    pub bsp1_lst: Vec<CBS_Point<T>>,
    pub config: CBSPointConfig,
    pub last_sure_pos: i64,
}

impl<T: Clone, L: Clone> CBSPointList<T, L> {
    pub fn new(bs_point_config: CBSPointConfig) -> Self {
        Self {
            lst: Vec::new(),
            bsp_dict: HashMap::new(),
            bsp1_lst: Vec::new(),
            config: bs_point_config,
            last_sure_pos: -1,
        }
    }

    pub fn __iter__(&self) -> std::slice::Iter<CBS_Point<T>> {
        self.lst.iter()
    }

    pub fn __len__(&self) -> usize {
        self.lst.len()
    }

    pub fn __getitem__(&self, index: usize) -> &CBS_Point<T> {
        &self.lst[index]
    }

    pub fn cal(&mut self, bi_list: L, seg_list: Vec<CSeg<T>>) {
        self.lst = self
            .lst
            .iter()
            .filter(|bsp| bsp.klu.idx <= self.last_sure_pos)
            .cloned()
            .collect();
        self.bsp_dict = self
            .lst
            .iter()
            .map(|bsp| (bsp.bi.get_end_klu().idx, bsp.clone()))
            .collect();
        self.bsp1_lst = self
            .bsp1_lst
            .iter()
            .filter(|bsp| bsp.klu.idx <= self.last_sure_pos)
            .cloned()
            .collect();

        self.cal_seg_bs1point(&seg_list, &bi_list);
        self.cal_seg_bs2point(&seg_list, &bi_list);
        self.cal_seg_bs3point(&seg_list, &bi_list);

        self.update_last_pos(&seg_list);
    }

    pub fn update_last_pos(&mut self, seg_list: &Vec<CSeg<T>>) {
        self.last_sure_pos = -1;
        for seg in seg_list.iter().rev() {
            if seg.is_sure {
                self.last_sure_pos = seg.end_bi.get_begin_klu().idx;
                return;
            }
        }
    }

    pub fn seg_need_cal(&self, seg: &CSeg<T>) -> bool {
        seg.end_bi.get_end_klu().idx > self.last_sure_pos
    }

    pub fn add_bs(
        &mut self,
        bs_type: BSP_TYPE,
        bi: T,
        relate_bsp1: Option<CBS_Point<T>>,
        is_target_bsp: bool,
        feature_dict: Option<HashMap<String, f64>>,
    ) {
        let is_buy = bi.is_down();
        if let Some(exist_bsp) = self.bsp_dict.get(&bi.get_end_klu().idx) {
            assert_eq!(exist_bsp.is_buy, is_buy);
            exist_bsp.add_another_bsp_prop(bs_type, relate_bsp1.map(Box::new));
            if let Some(feature_dict) = feature_dict {
                for (key, value) in feature_dict {
                    exist_bsp.add_feat(key, Some(value));
                }
            }
            return;
        }
        if !self
            .config
            .get_bs_config(is_buy)
            .target_types
            .contains(&bs_type)
        {
            return;
        }
        let bsp = CBS_Point::new(
            bi.clone(),
            is_buy,
            bs_type,
            relate_bsp1.map(Box::new),
            feature_dict,
        );
        if is_target_bsp {
            self.lst.push(bsp.clone());
            self.bsp_dict.insert(bi.get_end_klu().idx, bsp.clone());
        }
        if bs_type == BSP_TYPE::T1 || bs_type == BSP_TYPE::T1P {
            self.bsp1_lst.push(bsp);
        }
    }

    pub fn cal_seg_bs1point(&mut self, seg_list: &Vec<CSeg<T>>, bi_list: &L) {
        for seg in seg_list {
            if !self.seg_need_cal(seg) {
                continue;
            }
            self.cal_single_bs1point(seg, bi_list);
        }
    }

    pub fn cal_single_bs1point(&mut self, seg: &CSeg<T>, bi_list: &L) {
        let bsp_conf = self.config.get_bs_config(seg.is_down());
        let zs_cnt = seg.get_multi_bi_zs_cnt();
        let is_target_bsp = bsp_conf.min_zs_cnt <= 0 || zs_cnt >= bsp_conf.min_zs_cnt;
        if seg.zs_lst.len() > 0
            && !seg.zs_lst.last().unwrap().is_one_bi_zs()
            && ((seg.zs_lst.last().unwrap().bi_out.is_some()
                && seg.zs_lst.last().unwrap().bi_out.as_ref().unwrap().idx >= seg.end_bi.idx)
                || seg.zs_lst.last().unwrap().bi_lst.last().unwrap().idx >= seg.end_bi.idx)
            && seg.end_bi.idx - seg.zs_lst.last().unwrap().get_bi_in().idx > 2
        {
            self.treat_bsp1(seg, bsp_conf, is_target_bsp);
        } else {
            self.treat_pz_bsp1(seg, bsp_conf, bi_list, is_target_bsp);
        }
    }

    pub fn treat_bsp1(&mut self, seg: &CSeg<T>, bsp_conf: &CBSPointConfig, is_target_bsp: bool) {
        let last_zs = seg.zs_lst.last().unwrap();
        let (break_peak, _) = last_zs.out_bi_is_peak(seg.end_bi.idx);
        if bsp_conf.bs1_peak && !break_peak {
            return;
        }
        let (is_diver, divergence_rate) = last_zs.is_divergence(bsp_conf, Some(&seg.end_bi));
        if !is_diver {
            return;
        }
        let feature_dict = HashMap::from([
            ("divergence_rate".to_string(), divergence_rate),
            ("zs_cnt".to_string(), seg.zs_lst.len() as f64),
        ]);
        self.add_bs(
            BSP_TYPE::T1,
            seg.end_bi.clone(),
            None,
            is_target_bsp,
            Some(feature_dict),
        );
    }

    pub fn treat_pz_bsp1(
        &mut self,
        seg: &CSeg<T>,
        bsp_conf: &CBSPointConfig,
        bi_list: &L,
        is_target_bsp: bool,
    ) {
        let last_bi = &seg.end_bi;
        let pre_bi = &bi_list[last_bi.idx as usize - 2];
        if last_bi.seg_idx != pre_bi.seg_idx {
            return;
        }
        if last_bi.dir != seg.dir {
            return;
        }
        if last_bi.is_down() && last_bi._low() > pre_bi._low() {
            return;
        }
        if last_bi.is_up() && last_bi._high() < pre_bi._high() {
            return;
        }
        let in_metric = pre_bi.cal_macd_metric(bsp_conf.macd_algo, false);
        let out_metric = last_bi.cal_macd_metric(bsp_conf.macd_algo, true);
        let (is_diver, divergence_rate) = (
            out_metric <= bsp_conf.divergence_rate * in_metric,
            out_metric / (in_metric + 1e-7),
        );
        if !is_diver {
            return;
        }
        let feature_dict = HashMap::from([
            ("divergence_rate".to_string(), divergence_rate),
            ("bsp1_bi_amp".to_string(), last_bi.amp()),
        ]);
        self.add_bs(
            BSP_TYPE::T1P,
            last_bi.clone(),
            None,
            is_target_bsp,
            Some(feature_dict),
        );
    }

    pub fn cal_seg_bs2point(&mut self, seg_list: &Vec<CSeg<T>>, bi_list: &L) {
        let bsp1_bi_idx_dict: HashMap<i64, CBS_Point<T>> = self
            .bsp1_lst
            .iter()
            .map(|bsp| (bsp.bi.idx, bsp.clone()))
            .collect();
        for seg in seg_list {
            let config = self.config.get_bs_config(seg.is_down());
            if !config.target_types.contains(&BSP_TYPE::T2)
                && !config.target_types.contains(&BSP_TYPE::T2S)
            {
                continue;
            }
            self.treat_bsp2(seg, &bsp1_bi_idx_dict, seg_list, bi_list);
        }
    }

    pub fn treat_bsp2(
        &mut self,
        seg: &CSeg<T>,
        bsp1_bi_idx_dict: &HashMap<i64, CBS_Point<T>>,
        seg_list: &Vec<CSeg<T>>,
        bi_list: &L,
    ) {
        if !self.seg_need_cal(seg) {
            return;
        }
        let bsp_conf = self.config.get_bs_config(seg.is_down());
        let bsp1_bi = &seg.end_bi;
        let bsp1_bi_idx = bsp1_bi.idx;
        let real_bsp1 = bsp1_bi_idx_dict.get(&bsp1_bi_idx).cloned();
        if bsp1_bi_idx + 2 >= bi_list.len() as i64 {
            return;
        }
        let break_bi = &bi_list[bsp1_bi_idx as usize + 1];
        let bsp2_bi = &bi_list[bsp1_bi_idx as usize + 2];
        if bsp_conf.bsp2_follow_1 && !self.bsp_dict.values().any(|bsp| bsp.bi.idx == bsp1_bi_idx) {
            return;
        }
        let retrace_rate = bsp2_bi.amp() / break_bi.amp();
        let bsp2_flag = retrace_rate <= bsp_conf.max_bs2_rate;
        if bsp2_flag {
            let feature_dict = HashMap::from([
                ("bsp2_retrace_rate".to_string(), retrace_rate),
                ("bsp2_break_bi_amp".to_string(), break_bi.amp()),
                ("bsp2_bi_amp".to_string(), bsp2_bi.amp()),
            ]);
            self.add_bs(
                BSP_TYPE::T2,
                bsp2_bi.clone(),
                real_bsp1,
                true,
                Some(feature_dict),
            );
        } else if bsp_conf.bsp2s_follow_2 {
            return;
        }
        if !self
            .config
            .get_bs_config(seg.is_down())
            .target_types
            .contains(&BSP_TYPE::T2S)
        {
            return;
        }
        self.treat_bsp2s(
            seg_list,
            bi_list,
            bsp2_bi.clone(),
            break_bi.clone(),
            real_bsp1,
            bsp_conf,
        );
    }

    pub fn treat_bsp2s(
        &mut self,
        seg_list: &Vec<CSeg<T>>,
        bi_list: &L,
        bsp2_bi: T,
        break_bi: T,
        real_bsp1: Option<CBS_Point<T>>,
        bsp_conf: &CBSPointConfig,
    ) {
        let mut bias = 2;
        let mut _low = None;
        let mut _high = None;
        while bsp2_bi.idx + bias < bi_list.len() as i64 {
            let bsp2s_bi = &bi_list[(bsp2_bi.idx + bias) as usize];
            if bsp_conf.max_bsp2s_lv.is_some() && bias / 2 > bsp_conf.max_bsp2s_lv.unwrap() {
                break;
            }
            if bsp2s_bi.seg_idx != bsp2_bi.seg_idx
                && (bsp2s_bi.seg_idx < seg_list.len() as i64 - 1
                    || bsp2s_bi.seg_idx - bsp2_bi.seg_idx >= 2
                    || seg_list[bsp2_bi.seg_idx as usize].is_sure)
            {
                break;
            }
            if bias == 2 {
                if !has_overlap(
                    bsp2_bi._low(),
                    bsp2_bi._high(),
                    bsp2s_bi._low(),
                    bsp2s_bi._high(),
                ) {
                    break;
                }
                _low = Some(bsp2_bi._low().max(bsp2s_bi._low()));
                _high = Some(bsp2_bi._high().min(bsp2s_bi._high()));
            } else if !has_overlap(
                _low.unwrap(),
                _high.unwrap(),
                bsp2s_bi._low(),
                bsp2s_bi._high(),
            ) {
                break;
            }
            if bsp2s_break_bsp1(bsp2s_bi.clone(), break_bi.clone()) {
                break;
            }
            let retrace_rate =
                (bsp2s_bi.get_end_val() - break_bi.get_end_val()).abs() / break_bi.amp();
            if retrace_rate > bsp_conf.max_bs2_rate {
                break;
            }
            let feature_dict = HashMap::from([
                ("bsp2s_retrace_rate".to_string(), retrace_rate),
                ("bsp2s_break_bi_amp".to_string(), break_bi.amp()),
                ("bsp2s_bi_amp".to_string(), bsp2s_bi.amp()),
                ("bsp2s_lv".to_string(), (bias / 2) as f64),
            ]);
            self.add_bs(
                BSP_TYPE::T2S,
                bsp2s_bi.clone(),
                real_bsp1.clone(),
                true,
                Some(feature_dict),
            );
            bias += 2;
        }
    }

    pub fn cal_seg_bs3point(&mut self, seg_list: &Vec<CSeg<T>>, bi_list: &L) {
        let bsp1_bi_idx_dict: HashMap<i64, CBS_Point<T>> = self
            .bsp1_lst
            .iter()
            .map(|bsp| (bsp.bi.idx, bsp.clone()))
            .collect();
        for seg in seg_list {
            if !self.seg_need_cal(seg) {
                continue;
            }
            let config = self.config.get_bs_config(seg.is_down());
            if !config.target_types.contains(&BSP_TYPE::T3A)
                && !config.target_types.contains(&BSP_TYPE::T3B)
            {
                continue;
            }
            let bsp1_bi = &seg.end_bi;
            let bsp1_bi_idx = bsp1_bi.idx;
            let bsp_conf = self.config.get_bs_config(seg.is_down());
            let real_bsp1 = bsp1_bi_idx_dict.get(&bsp1_bi_idx).cloned();
            let next_seg_idx = seg.idx + 1;
            let next_seg = seg.next.as_ref();
            if bsp_conf.bsp3_follow_1
                && !self.bsp_dict.values().any(|bsp| bsp.bi.idx == bsp1_bi_idx)
            {
                continue;
            }
            if let Some(next_seg) = next_seg {
                self.treat_bsp3_after(
                    seg_list,
                    next_seg,
                    bsp_conf,
                    bi_list,
                    real_bsp1.clone(),
                    bsp1_bi_idx,
                    next_seg_idx,
                );
            }
            self.treat_bsp3_before(
                seg_list,
                seg,
                next_seg,
                bsp1_bi.clone(),
                bsp_conf,
                bi_list,
                real_bsp1.clone(),
                next_seg_idx,
            );
        }
    }

    pub fn treat_bsp3_after(
        &mut self,
        seg_list: &Vec<CSeg<T>>,
        next_seg: &CSeg<T>,
        bsp_conf: &CBSPointConfig,
        bi_list: &L,
        real_bsp1: Option<CBS_Point<T>>,
        bsp1_bi_idx: i64,
        next_seg_idx: i64,
    ) {
        let first_zs = next_seg.get_first_multi_bi_zs();
        if first_zs.is_none() {
            return;
        }
        let first_zs = first_zs.unwrap();
        if bsp_conf.strict_bsp3 && first_zs.get_bi_in().idx != bsp1_bi_idx + 1 {
            return;
        }
        if first_zs.bi_out.is_none()
            || first_zs.bi_out.as_ref().unwrap().idx + 1 >= bi_list.len() as i64
        {
            return;
        }
        let bsp3_bi = &bi_list[first_zs.bi_out.as_ref().unwrap().idx as usize + 1];
        if bsp3_bi.parent_seg.is_none() {
            if next_seg.idx != seg_list.len() as i64 - 1 {
                return;
            }
        } else if bsp3_bi.parent_seg.as_ref().unwrap().idx != next_seg.idx {
            if bsp3_bi.parent_seg.as_ref().unwrap().bi_list.len() >= 3 {
                return;
            }
        }
        if bsp3_bi.dir == next_seg.dir {
            return;
        }
        if bsp3_bi.seg_idx != next_seg_idx && next_seg_idx < seg_list.len() as i64 - 2 {
            return;
        }
        if bsp3_back2zs(bsp3_bi.clone(), first_zs.clone()) {
            return;
        }
        let bsp3_peak_zs = bsp3_break_zspeak(bsp3_bi.clone(), first_zs.clone());
        if bsp_conf.bsp3_peak && !bsp3_peak_zs {
            return;
        }
        let feature_dict = HashMap::from([
            (
                "bsp3_zs_height".to_string(),
                (first_zs.high - first_zs.low) / first_zs.low,
            ),
            ("bsp3_bi_amp".to_string(), bsp3_bi.amp()),
        ]);
        self.add_bs(
            BSP_TYPE::T3A,
            bsp3_bi.clone(),
            real_bsp1,
            true,
            Some(feature_dict),
        );
    }

    pub fn treat_bsp3_before(
        &mut self,
        seg_list: &Vec<CSeg<T>>,
        seg: &CSeg<T>,
        next_seg: Option<&CSeg<T>>,
        bsp1_bi: T,
        bsp_conf: &CBSPointConfig,
        bi_list: &L,
        real_bsp1: Option<CBS_Point<T>>,
        next_seg_idx: i64,
    ) {
        let cmp_zs = seg.get_final_multi_bi_zs();
        if cmp_zs.is_none() {
            return;
        }
        let cmp_zs = cmp_zs.unwrap();
        if bsp_conf.strict_bsp3
            && (cmp_zs.bi_out.is_none() || cmp_zs.bi_out.as_ref().unwrap().idx != bsp1_bi.idx)
        {
            return;
        }
        let end_bi_idx = cal_bsp3_bi_end_idx(next_seg);
        for bsp3_bi in bi_list.iter().skip(bsp1_bi.idx as usize + 2).step_by(2) {
            if bsp3_bi.idx > end_bi_idx {
                break;
            }
            if bsp3_bi.seg_idx != next_seg_idx && bsp3_bi.seg_idx < seg_list.len() as i64 - 1 {
                break;
            }
            if bsp3_back2zs(bsp3_bi.clone(), cmp_zs.clone()) {
                continue;
            }
            let feature_dict = HashMap::from([
                (
                    "bsp3_zs_height".to_string(),
                    (cmp_zs.high - cmp_zs.low) / cmp_zs.low,
                ),
                ("bsp3_bi_amp".to_string(), bsp3_bi.amp()),
            ]);
            self.add_bs(
                BSP_TYPE::T3B,
                bsp3_bi.clone(),
                real_bsp1.clone(),
                true,
                Some(feature_dict),
            );
            break;
        }
    }

    pub fn get_lastest_bsp_list(&self) -> Vec<CBS_Point<T>> {
        if self.lst.is_empty() {
            return Vec::new();
        }
        let mut sorted_lst = self.lst.clone();
        sorted_lst.sort_by(|a, b| b.bi.idx.cmp(&a.bi.idx));
        sorted_lst
    }
}

pub fn bsp2s_break_bsp1<T: Clone>(bsp2s_bi: &T, bsp2_break_bi: &T) -> bool {
    (bsp2s_bi.is_down() && bsp2s_bi._low() < bsp2_break_bi._low())
        || (bsp2s_bi.is_up() && bsp2s_bi._high() > bsp2_break_bi._high())
}

pub fn bsp3_back2zs<T: Clone>(bsp3_bi: &T, zs: &CZS<T>) -> bool {
    (bsp3_bi.is_down() && bsp3_bi._low() < zs.high) || (bsp3_bi.is_up() && bsp3_bi._high() > zs.low)
}

pub fn bsp3_break_zspeak<T: Clone>(bsp3_bi: &T, zs: &CZS<T>) -> bool {
    (bsp3_bi.is_down() && bsp3_bi._high() >= zs.peak_high)
        || (bsp3_bi.is_up() && bsp3_bi._low() <= zs.peak_low)
}

pub fn cal_bsp3_bi_end_idx<T: Clone>(seg: Option<&CSeg<T>>) -> i64 {
    if seg.is_none() {
        return i64::MAX;
    }
    let seg = seg.unwrap();
    if seg.get_multi_bi_zs_cnt() == 0 && seg.next.is_none() {
        return i64::MAX;
    }
    let mut end_bi_idx = seg.end_bi.idx - 1;
    for zs in &seg.zs_lst {
        if zs.is_one_bi_zs() {
            continue;
        }
        if zs.bi_out.is_some() {
            end_bi_idx = zs.bi_out.as_ref().unwrap().idx;
            break;
        }
    }
    end_bi_idx
}
