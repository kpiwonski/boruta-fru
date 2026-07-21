use pyo3::prelude::*;

#[pymodule(gil_used = false)]
#[pyo3(name = "_rust")]
mod pyfru {
    use boruta_arrow::Decision;
    use minarrow::{
        Array, CategoricalArray, FieldArray, FloatArray, NumericArray, StringArray, Table,
        TextArray,
    };
    use minarrow_pyo3::{PyArray, PyRecordBatch};
    use pyo3::{pyclass, pyfunction, pymethods};

    #[pyclass]
    pub struct BorutaRes(Vec<boruta_arrow::HitAggregator>, Vec<String>);

    #[pymethods]
    impl BorutaRes {
        fn final_decision(&self) -> PyRecordBatch {
            let decision: Vec<_> = self
                .0
                .iter()
                .map(|h| match h.decision {
                    Decision::Tentative => 0,
                    Decision::Confirmed => 1,
                    Decision::Rejected => 2,
                })
                .collect();
            let decision = FieldArray::from_arr(
                "decision",
                Array::from_categorical8(CategoricalArray::from_slices(
                    &decision,
                    &[
                        "Tentative".to_string(),
                        "Confirmed".to_string(),
                        "Rejected".to_string(),
                    ],
                )),
            );

            let colnames = FieldArray::from_arr(
                "column",
                Array::from_string64(StringArray::from_vec(
                    self.1.iter().map(|x| x.as_str()).collect::<Vec<_>>(),
                    None,
                )),
            );

            Table::new("final_decision".into(), vec![colnames, decision].into()).into()
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyfunction]
    #[pyo3(signature = (x, y, max_runs, pval_th, trees, tries, seed, threads=None))]
    pub fn boruta(
        x: PyRecordBatch,
        y: PyArray,
        max_runs: usize,
        pval_th: f64,
        trees: usize,
        tries: Option<usize>,
        seed: u64,
        threads: Option<usize>,
    ) -> BorutaRes {
        let x_df = x.into_inner();
        let col_names: Vec<_> = x_df.col_names().iter().map(|x| x.to_string()).collect();

        let y_array: Array = match y.into_inner().array {
            Array::TextArray(arr) => match arr {
                TextArray::Categorical8(x) => Into::<CategoricalArray<u64>>::into(&*x),
                TextArray::Categorical16(x) => Into::<CategoricalArray<u64>>::into(&*x),
                TextArray::Categorical32(x) => Into::<CategoricalArray<u64>>::into(&*x),
                TextArray::Categorical64(x) => (*x).clone(),
                _ => unreachable!("Decision is not categorical"),
            }
            .into(),
            Array::NumericArray(num) => match num {
                NumericArray::Int8(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::Int16(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::Int32(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::Int64(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::UInt8(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::UInt16(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::UInt32(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::UInt64(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::Float32(x) => Into::<FloatArray<f64>>::into(&*x),
                NumericArray::Float64(x) => (*x).clone(),
                _ => unreachable!("Decision is not numeric"),
            }
            .into(),
            _ => unreachable!("Unsupported decision type"),
        };

        let tries = tries.unwrap_or((x_df.n_cols() as f64).sqrt().round() as usize);
        let mut res: Vec<_> = boruta_arrow::boruta(
            x_df, y_array, max_runs, pval_th, trees, tries, seed, threads,
        )
        .collect();

        res.sort_by_key(|x| x.0);

        BorutaRes(res.iter().map(|x| x.1.clone()).collect(), col_names)
    }
}
