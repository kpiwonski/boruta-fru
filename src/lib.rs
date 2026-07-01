mod binom;

use fru_arrow::RandomForest;
use minarrow::{Array, ColumnSelection, FieldArray, RowSelection, Table};
use xrf::{Mask, RfRng};

use crate::binom::binom_cdf;

pub fn boruta(
    x: Table,
    y: Array,
    max_runs: usize,
    pval_th: f64,
    trees: usize,
    tries: usize,
    seed: u64,
    threads: Option<usize>,
) -> impl Iterator<Item = (usize, HitAggregator)> {
    let mut rng = RfRng::from_seed(seed, 1);

    let mut hits = vec![HitAggregator::new(); x.n_cols()];

    for run in 0..max_runs {
        let tentative_idxs: Vec<_> = hits
            .iter()
            .enumerate()
            .filter(|(_, d)| d.decision == Decision::Tentative)
            .map(|(i, _)| i)
            .collect();

        if tentative_idxs.len() == 0 {
            break;
        }

        let idxs: Vec<_> = hits
            .iter()
            .enumerate()
            .filter(|(_, d)| d.decision != Decision::Rejected)
            .map(|(i, _)| i)
            .collect();

        let mut xp = x.c(&*idxs).to_table();
        for i in 0..x.n_cols() {
            let mask = Mask::new_all(x.n_rows()).permute(&mut rng);
            let fa = FieldArray::from_arr("__shadow__", x.cols[i].r(&*mask).to_array()); // TODO name
            xp.add_col(fa);
        }

        let rf = RandomForest::fit(
            xp,
            y.clone(),
            trees,
            tries,
            false,
            true,
            false,
            rng.get_u64(),
            threads,
        );

        let max_shadow_imp = *rf
            .importance_raw(true)
            .iter()
            .filter(|(i, _)| *i >= idxs.len())
            .map(|(_, imp_val)| imp_val)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        for (i, imp_val) in rf.importance_raw(true) {
            if i < idxs.len() && max_shadow_imp < imp_val {
                hits[idxs[i]].ingest_hit(true);
            } else {
                hits[idxs[i]].ingest_hit(false);
            }
        }

        for &idx in tentative_idxs.iter() {
            hits[idx].decide(tentative_idxs.len(), pval_th);
        }
    }

    hits.into_iter().enumerate()
}

#[derive(Clone, Eq, PartialEq)]
pub enum Decision {
    Tentative,
    Confirmed,
    Rejected,
}

#[derive(Clone)]
pub struct HitAggregator {
    hits: usize,
    tries: usize,
    pub decision: Decision,
}

impl HitAggregator {
    fn new() -> Self {
        Self {
            hits: 0,
            tries: 0,
            decision: Decision::Tentative,
        }
    }

    fn decide(&mut self, tentative_num: usize, pval_th: f64) {
        if self.decision != Decision::Tentative {
            panic!("Cannot change decision when it is already confirmed or rejected");
        }

        let pval_rej = binom_cdf(self.hits as u64, self.tries as u64, 0.5);

        if pval_rej < pval_th / (tentative_num as f64) {
            self.decision = Decision::Rejected;
        }

        if self.hits > 0 {
            let pval_conf = binom_cdf((self.hits - 1) as u64, self.tries as u64, 0.5);
            if pval_conf > 1. - pval_th / (tentative_num as f64) {
                self.decision = Decision::Confirmed;
            }
        }
    }

    fn ingest_hit(&mut self, hitted: bool) {
        self.hits += hitted as usize;
        self.tries += 1;
    }
}
