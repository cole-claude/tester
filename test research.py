"""Run with:  python -m pytest -q   (or just: python test_research.py)"""
import math
import numpy as np
import pandas as pd
import research as R

P = dict(rev0=1000, m0=.25, mt=.25, wacc=.09, tg=.0, tax=.21, s2c=2.0, net_debt=0, shares=100)


def test_dcf_reconciles_to_perpetuity():
    # zero growth, constant margin: FCFF = rev*m*(1-t) forever, so EV = FCFF / wacc exactly
    d = R.dcf(g1=0, **P)
    expected = 1000 * .25 * (1 - .21) / .09 / 100
    assert math.isclose(d["per_share"], expected, rel_tol=1e-9)


def test_net_debt_and_shares_flow_through():
    base = R.dcf(g1=.05, **P)["per_share"]
    assert math.isclose(R.dcf(g1=.05, **{**P, "net_debt": 500})["per_share"], base - 5, rel_tol=1e-9)
    assert math.isclose(R.dcf(g1=.05, **{**P, "shares": 200})["per_share"], base / 2, rel_tol=1e-9)


def test_monotonic_and_deterministic():
    assert R.dcf(g1=.10, **P)["per_share"] > R.dcf(g1=.05, **P)["per_share"]
    assert R.dcf(g1=.05, **{**P, "wacc": .11})["per_share"] < R.dcf(g1=.05, **P)["per_share"]
    assert R.dcf(g1=.07, **P)["per_share"] == R.dcf(g1=.07, **P)["per_share"]


def test_reverse_dcf_round_trip():
    price = R.dcf(g1=.12, **{**P, "tg": .025})["per_share"]
    p = {k: v for k, v in P.items()}; p["tg"] = .025
    assert abs(R.reverse_dcf(price, **p) - .12) < 1e-6
    assert R.reverse_dcf(1e9, **p) is None


def test_negative_earnings_is_finite():
    d = R.dcf(g1=.2, **{**P, "m0": -.15, "mt": .15})
    assert np.isfinite(d["per_share"])


def test_probabilities_sum_to_one():
    for q in (-1, -.3, 0, .6, 1):
        assert math.isclose(sum(R.probs(q).values()), 1.0)
        assert all(.09 < v < .81 for v in R.probs(q).values())
    s = R.scenarios({**P, "g1": .08, "tg": .025}, .2)
    assert s["Bull"]["fv"] > s["Base"]["fv"] > s["Bear"]["fv"]


def test_justified_price_to_book():
    s = R.fin_scenarios(bvps=20, roe=.10, ke=.10, g=.03, q=0)
    assert math.isclose(s["Base"]["fv"], 20.0)  # ROE == cost of equity -> trades at book


def test_expected_returns():
    r = R.expected_returns(100, 100, .09)
    assert abs(r[5] - .09) < 1e-9 and abs(r[1] - .09) < 1e-9   # fairly priced: earn the cost of equity
    assert R.expected_returns(100, 150, .09)[3] > .09 > R.expected_returns(100, 70, .09)[3]


def test_verdict_gating_and_thresholds():
    assert R.verdict(.5, .04, .09)[0] == "Hold"            # great score, but return below hurdle
    assert R.verdict(.5, .15, .09)[0] == "Strong Buy"
    assert R.verdict(-.5, None, .09)[0] == "Strong Avoid"


def test_missing_data_not_fabricated():
    flags, m = R.forensic(None, None, None)
    assert flags == [] and m == {}
    sc = R.make_scores({}, None, 10, 0)
    assert sc["Expected return"] is None and sc["AI opportunity"] is None
    tot, _ = R.overall(sc, R.DEFAULT_W)
    assert np.isfinite(tot)


def test_contradictions_need_all_inputs():
    assert R.contradictions({}) == []
    assert R.contradictions({"tv_share": .8}) != []


def test_sensitivity_grid_center_matches_base():
    p = {**P, "g1": .06, "tg": .025}
    g = R.sensitivity(p, "wacc", "tg", [-.01, 0, .01], [-.005, 0, .005])
    assert math.isclose(g.iloc[1, 1], R.dcf(**p)["per_share"])


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
