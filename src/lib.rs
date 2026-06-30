use fru_arrow::RandomForest;
use minarrow::{Array, FieldArray, RowSelection, Table};
use xrf::{Mask, RfRng};

pub fn boruta(
    x: Table,
    y: Array,
    max_runs: usize,
    trees: usize,
    tries: usize,
    seed: u64,
    threads: Option<usize>,
) {
    let hits = vec![0usize; x.n_cols()];
    let mut rng = RfRng::from_seed(seed, 1);

    for run in 0..max_runs {
        // Create shadow columns
        let mut xp = x.clone();
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
        let imp = rf.importance_raw(true);
    }
}
