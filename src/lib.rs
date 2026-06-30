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
) -> {
    let mut hits = vec![0usize; x.n_cols()];
    let mut rng = RfRng::from_seed(seed, 1);

    let mut tentative: Vec<_> = (0..x.n_cols()).collect();
    let mut confirmed = vec![];
    let mut rejected: Vec<usize> = vec![];

    for run in 0..max_runs {
        if tentative.len() == 0 {
            break;
        }

        let idxs: Vec<_> = confirmed.iter().chain(tentative.iter()).copied().collect();
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
            seed, // TODO should be random int from rng
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
                hits[idxs[i]] += 1;
            }
        }

        let mut tentative_new = vec![];

        for &idx in tentative.iter() {
            let h = hits[idx];
            let pval_rej = binom_cdf(h as u64, run as u64, 0.5);
            let mut moved = false;

            if pval_rej < pval_th / (tentative.len() as f64) {
                rejected.push(idx);
                moved = true;
            }
            if hits[idx] > 0 {
                let pval_conf = binom_cdf((h - 1) as u64, run as u64, 0.5);
                if pval_conf > 1. - pval_th / (tentative.len() as f64) {
                    confirmed.push(idx.clone());
                    moved = true;
                }
            }

            if !moved {
                tentative_new.push(idx);
            }
        }
        tentative = tentative_new;
    }
}
