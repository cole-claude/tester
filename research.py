"""Deterministic research engine. No UI, no network: every number here is reproducible from its inputs."""
import numpy as np
import pandas as pd

DEFAULT_W = {"Business quality": .15, "Growth quality": .10, "Valuation attractiveness": .25,
             "Expected return": .25, "Downside risk": .15, "Financial strength": .10}


def getrow(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna().astype(float)
            if len(s):
                return s
    return None


# ---------------------------------------------------------------- valuation
def dcf(rev0, g1, m0, mt, wacc, tg, tax, s2c, net_debt, shares, years=10):
    """FCFF DCF. Growth fades linearly g1 -> tg; operating margin moves linearly m0 -> mt.
    Reinvestment = change in revenue / sales-to-capital. EBIT is GAAP, so stock comp is already a cost."""
    wacc = max(wacc, tg + .01)
    rev, pv, rows = rev0, 0.0, []
    for t in range(1, years + 1):
        g = g1 + (tg - g1) * (t - 1) / (years - 1)
        new = rev * (1 + g)
        m = m0 + (mt - m0) * t / years
        ebit = new * m
        fcff = (ebit * (1 - tax) if ebit > 0 else ebit) - (new - rev) / s2c
        pv += fcff / (1 + wacc) ** t
        rows.append({"Year": t, "Revenue": new, "Growth": g, "Op margin": m, "FCFF": fcff})
        rev = new
    roic = max(mt * (1 - tax) * s2c, wacc)  # terminal growth never creates value below the cost of capital
    nopat_t = max(rev * (1 + tg) * mt, 0) * (1 - tax)
    pv_tv = nopat_t * (1 - tg / roic) / (wacc - tg) / (1 + wacc) ** years
    ev = pv + pv_tv
    return {"per_share": (ev - net_debt) / shares, "ev": ev,
            "tv_share": pv_tv / ev if ev > 0 else float("nan"), "rows": pd.DataFrame(rows)}


def reverse_dcf(price, **p):
    """Growth rate that makes the DCF equal today's price (None if no solution in -25%..60%)."""
    f = lambda g: dcf(g1=g, **p)["per_share"] - price
    lo, hi = -.25, .60
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(lo) * f(mid) <= 0 else (mid, hi)
    return (lo + hi) / 2


def plausibility(g, hist_g):
    if g is None:
        return "n/a"
    if hist_g is not None and hist_g > .02:
        r = g / hist_g
        return "Conservative" if r <= .5 else "Reasonable" if r <= 1.0 else "Aggressive" if r <= 1.5 else "Implausible"
    return "Conservative" if g <= .03 else "Reasonable" if g <= .08 else "Aggressive" if g <= .15 else "Implausible"


def probs(q):
    """Scenario probabilities tilted by fundamentals score q in [-1, 1]; always sum to 1."""
    pb = float(np.clip(.25 + .10 * q, .10, .40))
    pr = float(np.clip(.25 - .10 * q, .10, .40))
    return {"Bull": pb, "Base": 1 - pb - pr, "Bear": pr}


def scenarios(p, q):
    bull = {**p, "g1": p["g1"] * 1.35 + .02, "mt": p["mt"] + max(.03, .15 * abs(p["mt"])),
            "wacc": p["wacc"] - .005, "tg": p["tg"] + .005}
    bear = {**p, "g1": p["g1"] * .45 - .01, "mt": p["mt"] - max(.04, .30 * abs(p["mt"])),
            "wacc": p["wacc"] + .01, "tg": max(p["tg"] - .005, 0), "s2c": p["s2c"] * .8}
    pr = probs(q)
    return {k: {"fv": max(dcf(**v)["per_share"], 0.0), "prob": pr[k], "p": v}
            for k, v in (("Bull", bull), ("Base", p), ("Bear", bear))}


def stress(p, price):
    cases = {"Recession (growth -8 pts, margin -25%)": {"g1": p["g1"] - .08, "mt": p["mt"] * .75 if p["mt"] > 0 else p["mt"]},
             "Rates +2 pts": {"wacc": p["wacc"] + .02},
             "Margin compression (-30%)": {"mt": p["mt"] * .7 if p["mt"] > 0 else p["mt"]},
             "Terminal growth -1 pt": {"tg": max(p["tg"] - .01, 0)}}
    return pd.DataFrame([{"Stress": k, "Fair value": max(dcf(**{**p, **v})["per_share"], 0),
                          "vs price": max(dcf(**{**p, **v})["per_share"], 0) / price - 1} for k, v in cases.items()])


def sensitivity(p, a, b, da, db):
    """Grid of fair value per share varying parameters a and b by offsets da x db."""
    return pd.DataFrame({f"{b} {p[b] + y:.3f}": [dcf(**{**p, a: p[a] + x, b: p[b] + y})["per_share"] for x in da]
                         for y in db}, index=[f"{a} {p[a] + x:.3f}" for x in da])


def fin_scenarios(bvps, roe, ke, g, q):
    """Justified price/book = (ROE - g) / (ke - g), for banks and similar balance-sheet businesses."""
    f = lambda r, k: max(bvps * (r - g) / (max(k, g + .01) - g), 0.0)
    pr = probs(q)
    return {"Bull": {"fv": f(roe * 1.1, ke - .01), "prob": pr["Bull"]}, "Base": {"fv": f(roe, ke), "prob": pr["Base"]},
            "Bear": {"fv": f(roe * .75, ke + .01), "prob": pr["Bear"]}}


def expected_returns(price, fv, ke, convergence_years=3):
    """Annualized return if price converges to fair value over `convergence_years`, with value compounding at ke."""
    out = {}
    for t in (1, 3, 5):
        w = min(t / convergence_years, 1.0)
        pt = price * (1 + ke) ** t * (max(fv, 1e-9) / price) ** w  # value compounds at ke; the gap closes gradually
        out[t] = (pt / price) ** (1 / t) - 1
    return out


# ---------------------------------------------------------------- forensic & logic checks
def forensic(inc, cf, bs):
    flags, m = [], {}
    g = lambda df, n: getrow(df, n)
    ni, rev = g(inc, ["Net Income"]), g(inc, ["Total Revenue", "Operating Revenue"])
    cfo, fcf, sbc = g(cf, ["Operating Cash Flow"]), g(cf, ["Free Cash Flow"]), g(cf, ["Stock Based Compensation"])
    ar, inv = g(bs, ["Accounts Receivable", "Receivables"]), g(bs, ["Inventory"])
    gw, eq, ta = g(bs, ["Goodwill"]), g(bs, ["Stockholders Equity", "Common Stock Equity"]), g(bs, ["Total Assets"])
    sh = g(bs, ["Ordinary Shares Number", "Share Issued"])
    gr = lambda s: s.iloc[0] / s.iloc[1] - 1 if s is not None and len(s) > 1 and s.iloc[1] > 0 else None
    if ni is not None and cfo is not None and ni.iloc[0] > 0:
        m["Cash conversion (CFO / net income)"] = cfo.iloc[0] / ni.iloc[0]
        if m["Cash conversion (CFO / net income)"] < .8:
            flags.append(("high", "Weak cash conversion", "Operating cash flow is well below reported earnings; investigate accruals."))
    if ni is not None and cfo is not None and ta is not None and ta.iloc[0] > 0:
        m["Accruals / assets"] = (ni.iloc[0] - cfo.iloc[0]) / ta.iloc[0]
        if m["Accruals / assets"] > .10:
            flags.append(("high", "High accruals", "Earnings rely heavily on non-cash items."))
    for nm, s in (("Receivables", ar), ("Inventory", inv)):
        if gr(s) is not None and gr(rev) is not None:
            m[f"{nm} growth minus revenue growth"] = gr(s) - gr(rev)
            if gr(s) - gr(rev) > .15:
                flags.append(("med", f"{nm} outpacing sales", "Can signal weak demand, channel stuffing or aggressive recognition (a reason to investigate, not proof)."))
    if sbc is not None and rev is not None:
        m["Stock comp / revenue"] = sbc.iloc[0] / rev.iloc[0]
        if fcf is not None:
            m["FCF after stock comp"] = fcf.iloc[0] - sbc.iloc[0]
        if m["Stock comp / revenue"] > .10:
            flags.append(("med", "Heavy stock-based compensation", "Reported free cash flow overstates what shareholders keep."))
    if gw is not None and eq is not None and eq.iloc[0] > 0:
        m["Goodwill / equity"] = gw.iloc[0] / eq.iloc[0]
        if m["Goodwill / equity"] > .5:
            flags.append(("med", "Goodwill is large vs equity", "Impairment risk if acquisitions underperform."))
    if sh is not None and len(sh) > 1:
        m["Share count change (YoY)"] = sh.iloc[0] / sh.iloc[1] - 1
        if m["Share count change (YoY)"] > .03:
            flags.append(("med", "Meaningful dilution", "Per-share value grows slower than the business."))
    return flags, m


def contradictions(c):
    """c: dict of optional numbers. Each rule fires only when every input it needs is present."""
    has = lambda *k: all(c.get(x) is not None for x in k)
    rules = [
        (has("rev_g", "margin_chg") and c["rev_g"] > .15 and c["margin_chg"] < -.02, "High revenue growth but operating margins are shrinking: growth may be bought, not earned."),
        (has("eps_g", "oi_g", "share_chg") and c["eps_g"] > .10 and c["oi_g"] < .03 and c["share_chg"] < -.02, "EPS growth is mostly buybacks; operating income is flat."),
        (has("cash_conv") and c["cash_conv"] < .7, "Reported earnings are not converting to cash."),
        (has("fwd_pe", "margin_peak") and c["fwd_pe"] < 12 and c["margin_peak"] and c["cyclical"], "Cheap forward P/E in a cyclical sector at peak margins: earnings may be unsustainable."),
        (has("ps", "gross_m") and c["ps"] < 2 and c["gross_m"] < .25, "Low P/S is not cheap when gross margins are structurally thin."),
        (has("fcf_yield", "rev_g") and c["fcf_yield"] > .06 and c["rev_g"] < 0, "High FCF yield alongside shrinking sales: the yield may be a melting ice cube."),
        (has("tv_share") and c["tv_share"] > .75, "Over 75% of DCF value sits in the terminal value: the result is very assumption-sensitive."),
        (has("quality", "exp5", "ke") and c["quality"] > .4 and c["exp5"] < c["ke"], "Excellent business, but the modeled return at today's price is below the required return."),
        (has("analyst", "rev_trend") and c["analyst"] > .5 and c["rev_trend"] < -.02, "Analysts are bullish while their own EPS estimates are being cut."),
    ]
    return [txt for ok, txt in rules if ok]


def validate(info, hist, price, now):
    w = []
    if hist is None or hist.empty:
        return ["No price history returned."]
    age = (now - hist.index[-1].tz_localize(None)).days if hist.index[-1].tzinfo else (now - hist.index[-1]).days
    if age > 5:
        w.append(f"Latest price bar is {age} days old (stale or market closed for a long period).")
    mc, sh = info.get("marketCap"), info.get("sharesOutstanding")
    if mc and sh and abs(mc / (price * sh) - 1) > .15:
        w.append("Market cap and shares x price disagree by >15% (multiple share classes or stale data).")
    if not info.get("sector"):
        w.append("No sector reported; generic thresholds used and industry logic limited.")
    return w


# ---------------------------------------------------------------- quant
def _naive(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
    return s


def quant(c, spy, rf=.04):
    c = _naive(c.dropna())
    if len(c) < 250:
        return {}
    r = c.pct_change().dropna()
    cagr = (c.iloc[-1] / c.iloc[0]) ** (252 / len(c)) - 1
    vol = r.std() * np.sqrt(252)
    dd = r[r < 0].std() * np.sqrt(252)
    out = {"Annualized return (history)": cagr, "Volatility": vol, "Max drawdown": (c / c.cummax() - 1).min(),
           "Downside deviation": dd, "Sharpe": (cagr - rf) / vol, "Sortino": (cagr - rf) / dd if dd else np.nan}
    if len(c) > 253:
        out["Momentum 12-1 months"] = c.iloc[-22] / c.iloc[-253] - 1
    if spy is not None and len(spy) > 250:
        s = _naive(spy["Close"].dropna()).pct_change().dropna()
        j = pd.concat([r, s], axis=1, join="inner").dropna()
        j.columns = ["a", "m"]
        if len(j) > 200:
            beta = j.cov().loc["a", "m"] / j.cov().loc["m", "m"]
            out.update({"Beta vs SPY": beta, "Correlation vs SPY": j.corr().loc["a", "m"],
                        "Alpha (annual, vs SPY)": (j.a.mean() - beta * j.m.mean()) * 252,
                        "6-mo relative strength vs SPY": (1 + j.a.tail(126)).prod() / (1 + j.m.tail(126)).prod() - 1})
    return out


# ---------------------------------------------------------------- scoring
def make_scores(cs, val, nmetrics, nflags):
    g = lambda k: float(cs.get(k, 0.0))
    clip = lambda x: float(np.clip(x, -1, 1))
    sc = {"Business quality": clip((g("Profitability") + g("Track record")) / 2),
          "Growth quality": clip(g("Growth") - .1 * min(nflags, 3)),
          "Financial strength": clip(g("Financial health")),
          "AI opportunity": None, "AI disruption risk": None}
    if val:
        rd = {"Conservative": 1, "Reasonable": .3, "Aggressive": -.5, "Implausible": -1}.get(val["plaus"], 0)
        sc["Valuation attractiveness"] = clip(.5 * clip(val["upside"] / .3) + .2 * g("Valuation") + .3 * rd)
        sc["Expected return"] = clip((val["exp5"] - val["ke"]) / .08)
        dd = max(0.0, 1 - val["bear"] / val["price"])
        sc["Downside risk"] = clip(.5 * (1 - 4 * dd) + .5 * g("Risk"))
        spread = (val["bull"] - val["bear"]) / max(val["base"], 1e-9)
    else:
        sc.update({"Valuation attractiveness": clip(g("Valuation")), "Expected return": None, "Downside risk": clip(g("Risk"))})
        spread = 1.0
    sc["Forecast confidence"] = float(np.clip(.4 * min(nmetrics / 45, 1) + .4 * (1 - min(spread / 2, 1)) + .2 * (1 - min(nflags, 5) / 5), 0, 1))
    return sc


def overall(sc, w):
    use = {k: v for k, v in w.items() if sc.get(k) is not None and v > 0}
    tot = sum(use.values())
    contrib = {k: sc[k] * v / tot for k, v in use.items()} if tot else {}
    return sum(contrib.values()), contrib


def verdict(score, exp5, hurdle):
    """Thresholds: >+0.40 Strong Buy, >+0.15 Buy, >-0.15 Hold, >-0.40 Avoid, else Strong Avoid.
    Buy ratings are capped at Hold when the modeled 5-yr return is below the required return."""
    v = ("Strong Buy" if score > .40 else "Buy" if score > .15 else "Hold" if score > -.15
         else "Avoid" if score > -.40 else "Strong Avoid")
    note = ""
    if exp5 is not None and exp5 < hurdle and v in ("Strong Buy", "Buy"):
        v, note = "Hold", f"Capped at Hold: modeled 5-yr return ({exp5:.1%}) is below the {hurdle:.1%} required return."
    return v, note
