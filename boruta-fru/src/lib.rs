//! Boruta-fru is an implementation of the Boruta algorithm for feature selection.
//! It leverages the `fru-arrow` Random Forest implementation to provide
//! fast computation of permutation-based feature importance.
mod binom;

use fru_arrow::RandomForest;
use log::info;
use minarrow::{Array, BooleanArray, ColumnSelection, FieldArray, RowSelection, Table};
use xrf::{Mask, RfRng};

use crate::binom::binom_cdf;

/// Main function for running the Boruta algorithm.
///
/// # Arguments
/// * `x` - minarrow `Table`.
/// * `y` - minarrow `Array`.
/// * `max_runs` - Maximum number of model iterations.
///   The process may stop early if all features are resolved as either confirmed or rejected.
///   If some features remain tentative, consider increasing this value.
/// * `pval_th` - Significance threshold (p-value).
///   The default value of 0.01 is recommended.
/// * `trees` - Number of trees in the forest.
/// * `tries` - Number of features to try at each split (often called `mtry`).
///   Must be greater than zero and less than or equal to the number of features.
///   A common default is the square root of the number of columns.
/// * `impute` - Controls how missing values are handled.
/// - If `true`, missing values are imputed by randomly sampling (with replacement)
///   from the non-null values in the same column. Imputation is performed before every
///   run of the forest, so the sampled values may differ each time.
/// - If `false`, the presence of any missing values will cause the code to panic.
/// * `seed` - Random seed used by the algorithm.
/// * `threads` - Number of threads to use. Must be greater than zero.
///   If `None`, all available CPU cores are used.
///
/// # Returns
/// An iterator over tuples `(usize, HitAggregator)`.
/// The first element is the index of the column.
/// The second element is an aggregator containing the model's decision.
///
/// # Notes
/// Boruta iteratively compares importances of attributes with importances of
/// shadow attributes, created by shuffling original ones. Attributes that have
/// significantly worst importance than shadow ones are being consecutively
/// dropped. On the other hand, attributes that are significantly better than shadows
/// are admitted to be Confirmed. Shadows are re-created in each iteration.
/// Algorithm stops when only Confirmed attributes are left, or when it reaches
/// max_runs importance source runs. If the second scenario occurs, some attributes
/// may be left without a decision. They are claimed Tentative.
#[allow(clippy::too_many_arguments)]
pub fn boruta(
    x: Table,
    y: Array,
    max_runs: usize,
    pval_th: f64,
    trees: usize,
    tries: usize,
    impute: bool,
    seed: u64,
    threads: Option<usize>,
) -> impl Iterator<Item = (usize, HitAggregator)> {
    let mut rng = RfRng::from_seed(seed, 1);

    let mut hits = vec![HitAggregator::new(); x.n_cols()];

    if !impute && x.cols.iter().any(|col| col.array.has_nulls()) {
        panic!("NA values are not supported without imputation");
    }

    for run in 0..max_runs {
        let tentative_idxs: Vec<_> = hits
            .iter()
            .enumerate()
            .filter(|(_, d)| d.decision == Decision::Tentative)
            .map(|(i, _)| i)
            .collect();

        if tentative_idxs.is_empty() {
            break;
        }

        info!(
            "Boruta iteration: {run}/{max_runs}. Tentative: {} Rejected: {} Confirmed: {}",
            hits.iter()
                .filter(|h| h.decision == Decision::Tentative)
                .count(),
            hits.iter()
                .filter(|h| h.decision == Decision::Rejected)
                .count(),
            hits.iter()
                .filter(|h| h.decision == Decision::Confirmed)
                .count()
        );

        let idxs: Vec<_> = hits
            .iter()
            .enumerate()
            .filter(|(_, d)| d.decision != Decision::Rejected)
            .map(|(i, _)| i)
            .collect();

        // Create table only with confirmed and tentative cols
        let mut xp = x.c(&*idxs).to_table();

        // Impute nan values
        if impute {
            for idx in 0..xp.n_cols() {
                let col = &mut xp.cols[idx];
                if col.array.has_nulls() {
                    let vals = (0..col.array.len())
                        .into_iter()
                        .map(|i| col.array.get_scalar(i));
                    let not_null_vals: Vec<_> = vals
                        .clone()
                        .filter_map(|opt| opt.filter(|x| !matches!(x, minarrow::Scalar::Null)))
                        .collect();
                    let null_idxs: Vec<usize> = vals
                        .enumerate()
                        .filter_map(|(i, opt)| {
                            matches!(opt, Some(minarrow::Scalar::Null)).then_some(i)
                        })
                        .collect();

                    if not_null_vals.is_empty() {
                        let arr = FieldArray::from_arr(
                            &*col.field.name,
                            Array::from_bool(BooleanArray::from_slice(
                                vec![false; col.len()].as_slice(),
                            )),
                        );
                        xp.cols[idx] = arr;
                        continue;
                    }

                    for idx in null_idxs {
                        let val = not_null_vals[rng.up_to(not_null_vals.len())].clone();
                        col.array.set(idx, val).unwrap();
                        col.refresh_null_count();
                    }
                }
            }
        }

        // Add shadow columns
        let mut num_shadow = xp.n_cols();
        while num_shadow < 5 {
            num_shadow *= 2;
        }

        for i in 0..num_shadow {
            let mask = Mask::new_all(xp.n_rows()).permute(&mut rng);
            let fa =
                FieldArray::from_arr("__shadow__", xp.cols[i % xp.n_cols()].r(&*mask).to_array()); // TODO name
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
            if i < idxs.len() {
                if max_shadow_imp < imp_val {
                    hits[idxs[i]].ingest_hit(true);
                } else {
                    hits[idxs[i]].ingest_hit(false);
                }
            }
        }

        for &idx in tentative_idxs.iter() {
            hits[idx].decide(tentative_idxs.len(), pval_th);
        }
    }

    hits.into_iter().enumerate()
}

/// The decision enum. Can be one of:
/// * `Tentative`
/// * `Confirmed`
/// * `Rejected`
#[derive(Clone, Eq, PartialEq)]
pub enum Decision {
    Tentative,
    Confirmed,
    Rejected,
}

/// The `HitAggregator` provides model results.
/// Currently only `decision` attribute is available.
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
