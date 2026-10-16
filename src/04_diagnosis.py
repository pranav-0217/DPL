"""Block 4: diagnose RR's middle overs and re-test earlier claims with bootstrap intervals."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RNG = np.random.default_rng(42)
N_BOOT = 10000
RR = "Rajasthan Royals"
PHASES = ["Powerplay", "Middle", "Death"]


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# --------------------------------------------------------------------------
def boot_ratio_diff(a, b, n=N_BOOT):
    """Bootstrap a difference of ratios from per-match (numerator, denominator) totals."""
    obs = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    i = RNG.integers(0, len(a), (n, len(a)))
    j = RNG.integers(0, len(b), (n, len(b)))
    bt = (a[i, 0].sum(1) / a[i, 1].sum(1)) - (b[j, 0].sum(1) / b[j, 1].sum(1))
    lo, hi = np.percentile(bt, [2.5, 97.5])
    p = 2 * min((bt >= 0).mean(), (bt <= 0).mean())
    return obs, lo, hi, p


def paired_diff(d, num, n=N_BOOT):
    """Bootstrap the RR-minus-others ratio gap, resampling whole matches with both innings together."""
    g = (d.assign(rr=d.team_bat == RR, one=1).groupby(["p_match", "rr"], observed=True)
           .agg(x=(num, "sum"), b=("one", "sum")).unstack("rr", fill_value=0))
    a_x, a_b, o_x, o_b = (g[(c, k)].values.astype(float) for k in (True, False) for c in ("x", "b"))
    ra, rb = a_x.sum() / a_b.sum(), o_x.sum() / o_b.sum()
    i = RNG.integers(0, len(g), (n, len(g)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bt = a_x[i].sum(1) / a_b[i].sum(1) - o_x[i].sum(1) / o_b[i].sum(1)
    bt = bt[~np.isnan(bt)]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return ra - rb, lo, hi, 2 * min((bt >= 0).mean(), (bt <= 0).mean()), ra, rb


def per_match(df, num, den, by="p_match"):
    g = df.groupby(by, observed=True).agg(n=(num, "sum"), d=(den, "sum"))
    g = g[g.d > 0]
    return g[["n", "d"]].values.astype(float)


def verdict(lo, hi):
    return "SIGNIFICANT" if lo * hi > 0 else "not significant"


# --------------------------------------------------------------------- Q1
def passive_or_beaten(df):
    rule("Q1  PASSIVE OR BEATEN? Decomposing the middle overs")
    print("  Split strike rate into how often a batter is in control and what they score when they are.\n")

    mid = df[(df.phase == "Middle") & (df.control.notna()) & (df.ballfaced == 1)]
    rows = []
    for y in [2023, 2024]:
        d = mid[mid.year == y]
        print(f"  -- {y} middle overs --")
        print(f"  {'team':30s} {'ctrl%':>7s} {'RPB|ctrl':>9s} {'RPB|beat':>9s} {'SR':>7s}")
        for t, g in d.groupby("team_bat", observed=True):
            c = g[g.control == 1]
            nc = g[g.control == 0]
            r = {"year": y, "team": t,
                 "ctrl_pct": 100 * g.control.mean(),
                 "rpb_in_control": c.batruns.sum() / len(c),
                 "rpb_beaten": nc.batruns.sum() / len(nc),
                 "sr": 100 * g.batruns.sum() / len(g)}
            rows.append(r)
        tab = pd.DataFrame([r for r in rows if r["year"] == y]).sort_values("rpb_in_control",
                                                                           ascending=False)
        for _, r in tab.iterrows():
            mark = "  <-- RR" if r.team == RR else ""
            print(f"  {r.team:30s} {r.ctrl_pct:7.1f} {r.rpb_in_control:9.3f} "
                  f"{r.rpb_beaten:9.3f} {r.sr:7.1f}{mark}")
        print()

    out = pd.DataFrame(rows)
    print("  Test: RR's runs per controlled ball vs the other nine teams, middle overs.\n")
    res = []
    for y in [2023, 2024, None]:
        d = mid if y is None else mid[mid.year == y]
        o, lo, hi, p, ra, rb = paired_diff(d[d.control == 1], "batruns")
        lab = str(y) if y else "POOLED"
        print(f"    {lab:7s} RR {ra:.3f}  others {rb:.3f}  diff {o:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  {verdict(lo, hi)}")
        res.append({"test": "runs per controlled ball", "season": lab, "rr": ra, "others": rb, "diff": o,
                    "ci_lo": lo, "ci_hi": hi, "p_value": p, "significant": bool(lo * hi > 0)})

    print("\n  Same test for how OFTEN RR were in control:")
    for y in [2023, 2024, None]:
        d = mid if y is None else mid[mid.year == y]
        o, lo, hi, p, ra, rb = paired_diff(d, "control")
        lab = str(y) if y else "POOLED"
        print(f"    {lab:7s} RR {100*ra:.1f}%  others {100*rb:.1f}%  diff {100*o:+.1f}pp "
              f"[{100*lo:+.1f}, {100*hi:+.1f}]  p={p:.3f}  {verdict(lo, hi)}")
        res.append({"test": "control frequency", "season": lab, "rr": ra, "others": rb, "diff": o,
                    "ci_lo": lo, "ci_hi": hi, "p_value": p, "significant": bool(lo * hi > 0)})
    return out, pd.DataFrame(res)


# --------------------------------------------------------------------- Q2
def spin_or_pace(df):
    rule("Q2  SPIN OR PACE? Where the middle-overs gap comes from")
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1)
             & df.bowl_kind.isin(["spin bowler", "pace bowler"])]
    res = []
    for kind in ["spin bowler", "pace bowler"]:
        print(f"\n  -- vs {kind} (middle overs) --")
        for y in [2023, 2024, None]:
            d = mid[mid.bowl_kind == kind]
            if y:
                d = d[d.year == y]
            o, lo, hi, p, ra, rb = paired_diff(d, "batruns")
            lab = str(y) if y else "POOLED"
            print(f"    {lab:7s} RR SR {100*ra:6.1f}  others {100*rb:6.1f}  diff {100*o:+6.1f} "
                  f"[{100*lo:+.1f}, {100*hi:+.1f}]  p={p:.3f}  {verdict(lo, hi)}")
            res.append({"bowl_kind": kind, "season": lab, "diff_sr": 100 * o,
                        "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p_value": p,
                        "significant": bool(lo * hi > 0)})

    print("\n  How much spin does each side face in the middle overs?")
    sh = (mid.groupby(["year", "team_bat"], observed=True)
            .apply(lambda g: 100 * (g.bowl_kind == "spin bowler").mean(), include_groups=False)
            .rename("spin_share_pct").reset_index())
    for y in [2023, 2024]:
        s = sh[sh.year == y].sort_values("spin_share_pct", ascending=False)
        line = "  ".join(f"{r.team_bat.split()[0]} {r.spin_share_pct:.0f}%" for _, r in s.iterrows())
        print(f"    {y}: {line}")
    return pd.DataFrame(res)


# --------------------------------------------------------------------- Q3
def who(df):
    rule("Q3  WHO? Player contributions with empirical-Bayes shrinkage")
    print("  Player rates are shrunk toward the league mean; small samples shrink most.\n")

    mid = df[(df.phase == "Middle") & (df.ballfaced == 1)]
    out = []
    for y in [2023, 2024]:
        d = mid[mid.year == y]
        g = d.groupby("bat", observed=True).agg(
            balls=("batruns", "size"), runs=("batruns", "sum"),
            team=("team_bat", "first"), ctrl=("control", "mean"))
        g = g[g.balls >= 20]
        g["rpb"] = g.runs / g.balls

        grand = d.batruns.sum() / len(d)
        sigma2 = d.batruns.var()                       # per-ball variance
        total_var = g.rpb.var()
        mean_samp_var = (sigma2 / g.balls).mean()
        tau2 = max(total_var - mean_samp_var, 1e-9)    # between-player variance
        g["B"] = tau2 / (tau2 + sigma2 / g.balls)
        g["rpb_shrunk"] = g.B * g.rpb + (1 - g.B) * grand
        g["sr_raw"] = 100 * g.rpb
        g["sr_shrunk"] = 100 * g.rpb_shrunk
        g["year"] = y

        print(f"  -- {y} middle overs --  league mean {100*grand:.1f} SR, "
              f"tau={100*np.sqrt(tau2):.1f}, n>=20 balls, {len(g)} players")
        rr = g[g.team == RR].sort_values("sr_shrunk", ascending=False)
        print(f"  {'player':22s} {'balls':>6s} {'raw SR':>7s} {'shrunk':>7s} "
              f"{'shrink':>7s} {'ctrl%':>6s} {'lg rank':>8s}")
        g_sorted = g.sort_values("sr_shrunk", ascending=False).reset_index()
        rank = {r.bat: i + 1 for i, r in g_sorted.iterrows()}
        for p, r in rr.iterrows():
            print(f"  {p:22s} {int(r.balls):6d} {r.sr_raw:7.1f} {r.sr_shrunk:7.1f} "
                  f"{100*(1-r.B):6.0f}% {100*r.ctrl:6.1f} {rank[p]:5d}/{len(g)}")
        print()
        out.append(g.reset_index())
    return pd.concat(out)


# --------------------------------------------------------------------- Q4
def retest(df, matches):
    rule("Q4  RE-TESTING CLAIMS MADE DURING EXPLORATION")
    print("  Two exploratory single-season claims, tested and reported either way.\n")

    res = []
    # ---- Claim A: RR's 2024 death-overs batting collapsed late in the season
    print("  CLAIM A: RR's 2024 death-overs SR fell from 197.8 (first 9 matches) to 140.2 (last 6).\n")
    rr24 = matches[(matches.year == 2024) &
                   ((matches.team_a == RR) | (matches.team_b == RR))].sort_values("match_date")
    order = {m: i + 1 for i, m in enumerate(rr24.p_match)}
    d = df[(df.year == 2024) & (df.team_bat == RR) & (df.phase == "Death")
           & (df.ballfaced == 1)].copy()
    d["idx"] = d.p_match.map(order)
    early = d[d.idx <= 9]
    late = d[d.idx > 9]
    a = per_match(early.assign(one=1), "batruns", "one")
    b = per_match(late.assign(one=1), "batruns", "one")
    o, lo, hi, p = boot_ratio_diff(a, b)
    print(f"    first 9 matches : SR {100*a[:,0].sum()/a[:,1].sum():.1f}  ({len(a)} innings)")
    print(f"    last 6 matches  : SR {100*b[:,0].sum()/b[:,1].sum():.1f}  ({len(b)} innings)")
    print(f"    difference      : {100*o:+.1f}  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]  "
          f"p={p:.3f}  -> {verdict(lo, hi)}")
    res.append({"claim": "2024 death-overs late collapse", "diff": 100 * o,
                "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p_value": p,
                "significant": bool(lo * hi > 0)})

    # The split was chosen after seeing the data, so test every possible split point.
    print("\n    The split was chosen after seeing the data, so every split point is tested by permutation.\n")

    inn = pd.concat([early, late])
    per = (inn.groupby("idx", observed=True)
              .agg(r=("batruns", "sum"), b=("batruns", "size"))
              .sort_index())
    r_, b_ = per.r.values.astype(float), per.b.values.astype(float)
    n_inn = len(r_)
    MIN_SIDE = 4
    splits = range(MIN_SIDE, n_inn - MIN_SIDE + 1)

    def max_break(r_v, b_v):
        best = 0.0
        for k in splits:
            d_ = (r_v[:k].sum() / b_v[:k].sum()) - (r_v[k:].sum() / b_v[k:].sum())
            best = max(best, abs(d_))
        return best

    obs_max = max_break(r_, b_)
    perm = np.empty(N_BOOT)
    for t_ in range(N_BOOT):
        o_ = RNG.permutation(n_inn)
        perm[t_] = max_break(r_[o_], b_[o_])
    p_corr = (perm >= obs_max).mean()
    print(f"    Innings available          : {n_inn}")
    print(f"    Candidate split points     : {len(list(splits))} (min {MIN_SIDE} innings a side)")
    print(f"    Largest break found        : {100*obs_max:.1f} SR points")
    print(f"    Permutation p (corrected)  : {p_corr:.4f}  -> "
          f"{'SIGNIFICANT' if p_corr < 0.05 else 'not significant'}")
    print(f"    Verdict: a break this large appears by chance {100*p_corr:.1f}% of the time, so the drop is not supported.\n")
    res.append({"claim": "2024 death collapse (permutation-corrected)",
                "diff": 100 * obs_max, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": p_corr, "significant": bool(p_corr < 0.05)})

    # ---- Claim B: RR underperform against spin
    print("  CLAIM B: RR scored 129.6 against spin in 2024 vs a league 141.4 (all phases).\n")
    d = df[(df.bowl_kind == "spin bowler") & (df.ballfaced == 1)]
    for y in [2023, 2024, None]:
        s = d if y is None else d[d.year == y]
        o, lo, hi, p, ra, rb = paired_diff(s, "batruns")
        lab = str(y) if y else "POOLED"
        print(f"    {lab:7s} RR {100*ra:6.1f}  others {100*rb:6.1f}  diff {100*o:+6.1f} "
              f"[{100*lo:+.1f}, {100*hi:+.1f}]  p={p:.3f}  {verdict(lo, hi)}")
        res.append({"claim": f"vs spin all phases {lab}", "diff": 100 * o,
                    "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p_value": p,
                    "significant": bool(lo * hi > 0)})
    return pd.DataFrame(res)


def main():
    df = pd.read_parquet(PROC / "clean.parquet")
    matches = pd.read_parquet(PROC / "matches.parquet")
    print(f"Loaded {len(df):,} deliveries, {len(matches)} matches")

    ctrl, ctrl_test = passive_or_beaten(df)
    sp = spin_or_pace(df)
    players = who(df)
    rt = retest(df, matches)

    rule("WRITING OUTPUT")
    for obj, name in [(ctrl, "diagnosis_control.csv"), (ctrl_test, "diagnosis_control_test.csv"),
                      (sp, "diagnosis_spin_pace.csv"), (players, "diagnosis_players.csv"),
                      (rt, "diagnosis_retest.csv")]:
        obj.to_csv(PROC / name, index=False)
        print(f"  {name:34s} {(PROC / name).stat().st_size / 1024:>7.1f} KB")
    print("\nBlock 4 complete.")


if __name__ == "__main__":
    main()
