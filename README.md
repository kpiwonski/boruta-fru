# Boruta Fru

[![PyPI Version](https://img.shields.io/pypi/v/boruta-fru)](https://pypi.org/project/boruta-fru/)
[![Crates.io Version](https://img.shields.io/crates/v/boruta-fru)](https://crates.io/crates/boruta-fru)

[Boruta-fru docs](https://kpiwonski.github.io/boruta-fru/) |
[R version](https://cran.r-project.org/web/packages/Boruta/index.html)

Boruta-fru is a canonical implementation of the [Boruta](iwww.jstatsoft.org/article/view/v036i11)
all-relevant feature selection algorithm for Python.
This package is coauthored by the original [R package](https://cran.r-project.org/web/packages/Boruta/index.html)
author Miron Kursa, with a goal to resemble the original implementation as close as possible.

Boruta-fru is built around [fru](github.com/kpiwonski/fru-arrow), a scalable Random Forest implementation featuring a novel, [highly-optimised algorithm](https://dx.doi.org/10.2139/ssrn.6864991) for calculating permutational importance; this way it doesn't need to compromise on using Gini importance for speed.
By using the [Arrow PyCapsule](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) underneath, it flawlessly integrates with any data frame library that supports it; that includes ``polars``, ``pandas`` and ``pyarrow``.


## What is Boruta?

Boruta works in a standard supervised learning setup; it expects a data set of observations described by numeric or categorical features and a single decision, and aims to find a subset of features that are relevant to the decision.
This is achieved using Random Forest importance as a relevance proxy and adding synthetic irrelevant features, *shadows*, as a base for iteratively applied permutational test.

Thanks to this construction, Boruta is an all relevant method, i.e., it doesn't remove remove weakly relevant (redundant) features, which is crucial for interpretability, but it is also more stable and allows for building more robust models.


## Boruta-fru versus borutapy

Borutapy is the first implementation of Boruta for Python, and is a part of scikit-learn contrib; it uses, by default, the scikit version of Random Forest, which is noticably slower than fru.
This way, Boruta-fru can substantially outperform Borutapy, being typically anywhere from a few time to several thousand times faster.
The plot below illustrates this difference for 3 datasets.

![Compare to borutapy](https://raw.githubusercontent.com/kpiwonski/boruta-fru/refs/heads/main/plt_cmp_borutapy.png)

Moreover, borutapy introduces several differences from the R package that may influence the results; boruta-fru follows the R package in these regards.
In particular, boruta-py:
- defaults to scikit Random Forest, which uses Gini importance, while original implementation defaults to the permutation importance;
- uses FDR correction by default, while it should apply Bonferroni correction;
- reports only confirmed features, while it should be also possible to distinguish between tentative and rejected ones;
- uses 0.05 default p-value cutoff, while it should use 0.01;
- handles missing values with through the Random Forest engine, while it should use transdapters.

## Boruta-fru versus R Boruta

As mentioned before, Boruta-fru was made to be as compatible with the R package as possible; still, there are some caveats.
Both implementations are based on fru, but it is tightly integrated Boruta-fru, while it can be swapped with other implementations in R Boruta.
Following this, Boruta-fru has no support for transdapters; the functionality of the most popular one, impute transdapter that allows for processing data with missing values, is implemented via the ``impute`` flag.

Due to a fact that Boruta is a stochastic algorithm and differences in how random seeds are handled in Python and R, it is not possible to exactly reproduce R result with Python and vice-versa, even with common random seeds.
The results should be asymptotically identical, though.
