import numpy as np
import pandas as pd
import boruta_fru
import pytest

rng = np.random.default_rng(seed=42)


@pytest.fixture
def table_0_1_10ft():
    # Generate random 0s and 1s
    num_rows = 100
    num_cols = 10
    data = {f"col{i + 1}": rng.integers(0, 2, size=num_rows) for i in range(num_cols)}
    df = pd.DataFrame(data)
    y = pd.Categorical(df.iloc[:, 0], categories=[0, 1], ordered=False)
    y = y.rename_categories({0: "No", 1: "Yes"})
    y = pd.Series(y)
    return (df, y)


def test_boruta_x0(table_0_1_10ft):
    b = boruta_fru.Boruta(tries=3, seed=0)
    b.fit(table_0_1_10ft[0], table_0_1_10ft[1])
    assert b.final_decision()[0] == "Confirmed"
    assert all(b.final_decision()[1:] == "Rejected")


def test_boruta_x0nx1(table_0_1_10ft):
    b = boruta_fru.Boruta(tries=3, seed=0)
    x = table_0_1_10ft[0]
    b.fit(x, x.iloc[:, 0] & x.iloc[:, 1])
    assert b.final_decision()[0] == "Confirmed"
    assert b.final_decision()[1] == "Confirmed"
    assert all(b.final_decision()[2:] == "Rejected")


def test_boruta_rand_decision(table_0_1_10ft):
    b = boruta_fru.Boruta(tries=3, seed=1)
    x = table_0_1_10ft[0]
    y = pd.Series(rng.integers(0, 2, size=100))
    b.fit(x, y)
    assert all(b.final_decision() == "Rejected")


def test_rf_cls_0_1_3ft_imp_pycapsule(table_0_1_10ft):
    X, y = table_0_1_10ft
    b = boruta_fru.Boruta(tries=3, seed=0)
    b.fit(X, y)
    res = b.final_decision(to_pycapsule=True)
    res = pd.DataFrame.from_arrow(res)
    assert list(res.columns) == ["column", "decision"]
    assert list(res["column"]) == list(["col" + str(i + 1) for i in range(10)])
    assert res["decision"][0] == "Confirmed"
    assert all(res["decision"].iloc[1:] == "Rejected")
