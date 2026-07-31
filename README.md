# Boruta Fru

Boruta-fru is a canonical implementation of the [Boruta](iwww.jstatsoft.org/article/view/v036i11)
all-relevant feature selection  algorithm.
This package is coauthored by the original [R package](https://cran.r-project.org/web/packages/Boruta/index.html)
author Miron Kursa, so the package resambles original implementation as close as possible.
It uses Arrow PyCapsule underneath, making integration with any library that supports
it - ``polars``, ``pandas``, ``pyarrow`` straightforward.
Thanks to our improvements in the [fru](github.com/kpiwonski/fru-arrow) Random Forest package, this package uses
efficient permutation importance calculation.

## Boruta-fru versus borutapy

The main difference between boruta-fru and borutapy is its speed. 
borutapy uses scikit version of Random Forest underneath, while boruta-fru uses efficient implementation
from the fru Random Forest package.
Boruta-fru is typically anywhere from a few time to several thousand times faster than borutapy.
The plot below illustrates this difference for 2 datasets.




Most other differences between borutapy and boruta-fru boils down to implementation details:
- borutapy uses impurity importance, while boruta-fru uses permutation importance
- borutapy uses by default FDR correcation, while boruta-fru uses Bonferoni correction
- borutapy is concentrated on confirming features. It is not possible to distinguish between tentative and rejected with borutapy.
- borutapy uses by default 0.05 pvalue, while boruta-fru uses 0.01
