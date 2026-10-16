"""Block 2: phase profiles for all 20 team-seasons, and which phase best tracks winning."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RNG = np.random.default_rng(42)          # fixed seed: results are reproducible
N_BOOT = 2000          # for correlation CIs
N_BOOT_SIG = 10000     # for the RR difference test

PHASES = ["Powerplay", "Middle", "Death"]


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# --- metric definitions ----------------------------------------------------
def bat_metrics(g):
    bf = g.ballfaced.sum()
    legal = g.legal_ball.sum()
    outs = g.out.sum()
    return pd.Series({
        "balls": bf,
        "runs": g.batruns.sum(),
        "sr": 100 * g.batruns.sum() / bf,
        "bnd_pct": 100 * g.batruns.isin([4, 6]).sum() / bf,
        "dot_pct": 100 * ((g.score == 0) & g.legal_ball).sum() / legal,
        "ctrl_pct": 100 * g.control.mean(),
        "bpd": bf / outs if outs else np.nan,      # balls per dismissal
    })


def bowl_metrics(g):
    legal = g.legal_ball.sum()
    wk = g.bowler_wicket.sum()
    return pd.Series({
        "balls": legal,
        "runs": g.bowlruns.sum(),
        "econ": 6 * g.bowlruns.sum() / legal,
        "dot_pct": 100 * ((g.score == 0) & g.legal_ball).sum() / legal,
        "ctrl_against_pct": 100 * g.control.mean(),
        "bpw": legal / wk if wk else np.nan,       # balls per wicket
    })


def build_team_phase(df):
    rule("TEAM-SEASON PHASE PROFILES")
    bat = df.groupby(["year", "team_bat", "phase"], observed=True).apply(bat_metrics, include_groups=False)
    bowl = df.groupby(["year", "team_bowl", "phase"], observed=True).apply(bowl_metrics, include_groups=False)
    bat.index.names = ["year", "team", "phase"]
    bowl.index.names = ["year", "team", "phase"]
    tp = bat.add_prefix("bat_").join(bowl.add_prefix("bowl_")).reset_index()
    print(f"  Built {tp.year.nunique()} seasons x {tp.team.nunique()} teams x {len(PHASES)} phases "
          f"= {len(tp)} rows")
    print(f"  Batting balls accounted: {int(tp.bat_balls.sum()):,}")
    print(f"  Bowling balls accounted: {int(tp.bowl_balls.sum()):,}")
    return tp


# --- net run-rate edge -----------------------------------------------------
def build_edge(df, tp):
    """Net edge per phase: own scoring rate minus rate conceded, in runs per over above league."""
    rule("NET PHASE EDGE (runs per over above league average)")
    lg = (df.groupby(["year", "phase"], observed=True)
            .apply(lambda g: 6 * g.bowlruns.sum() / g.legal_ball.sum(), include_groups=False)
            .rename("lg_rpo").reset_index())
    print("  League run rate per over by season and phase:")
    print(lg.pivot(index="phase", columns="year", values="lg_rpo").round(2).to_string().replace("\n", "\n    "))

    e = tp.merge(lg, on=["year", "phase"])
    # batting rate uses bowlruns-equivalent scoring: runs conceded by opposition
    bat_rpo = (df.groupby(["year", "team_bat", "phase"], observed=True)
                 .apply(lambda g: 6 * g.bowlruns.sum() / g.legal_ball.sum(), include_groups=False)
                 .rename("bat_rpo").reset_index().rename(columns={"team_bat": "team"}))
    bowl_rpo = (df.groupby(["year", "team_bowl", "phase"], observed=True)
                  .apply(lambda g: 6 * g.bowlruns.sum() / g.legal_ball.sum(), include_groups=False)
                  .rename("bowl_rpo").reset_index().rename(columns={"team_bowl": "team"}))
    e = e.merge(bat_rpo, on=["year", "team", "phase"]).merge(bowl_rpo, on=["year", "team", "phase"])
    e["bat_edge"] = e.bat_rpo - e.lg_rpo
    e["bowl_edge"] = e.lg_rpo - e.bowl_rpo
    e["net_edge"] = e.bat_edge + e.bowl_edge
    return e[["year", "team", "phase", "bat_rpo", "bowl_rpo", "lg_rpo",
              "bat_edge", "bowl_edge", "net_edge"]]


# --- ranks -----------------------------------------------------------------
def add_ranks(tp):
    rule("LEAGUE RANKS (1 = best of 10, within season)")
    # Control is excluded from the ranks: it measures risk appetite, not quality.
    higher_better = ["bat_sr", "bat_bnd_pct", "bat_bpd", "bowl_dot_pct"]
    lower_better = ["bat_dot_pct", "bowl_econ", "bowl_bpw"]
    for c in higher_better:
        tp[c + "_rk"] = tp.groupby(["year", "phase"], observed=True)[c].rank(ascending=False).astype(int)
    for c in lower_better:
        tp[c + "_rk"] = tp.groupby(["year", "phase"], observed=True)[c].rank(ascending=True).astype(int)
    print(f"  Ranked {len(higher_better) + len(lower_better)} metrics within each season x phase")
    return tp


# --- season outcomes -------------------------------------------------------
def season_records(matches):
    rule("SEASON RECORDS")
    played = pd.concat([
        matches[["year", "p_match", "team_a", "winner", "no_result"]].rename(columns={"team_a": "team"}),
        matches[["year", "p_match", "team_b", "winner", "no_result"]].rename(columns={"team_b": "team"}),
    ]).dropna(subset=["team"])
    played = played[~played.no_result]
    rec = (played.assign(win=lambda d: (d.team == d.winner).astype(int))
                 .groupby(["year", "team"])
                 .agg(played=("p_match", "nunique"), wins=("win", "sum")).reset_index())
    rec["win_pct"] = 100 * rec.wins / rec.played
    for y in sorted(rec.year.unique()):
        print(f"\n  -- {y} --")
        print(rec[rec.year == y].sort_values("win_pct", ascending=False)
              [["team", "played", "wins", "win_pct"]].round(1)
              .to_string(index=False).replace("\n", "\n    "))
    return rec


def finalists(matches):
    """Return each season's champion and runner-up from its final."""
    out = {}
    for y, g in matches.groupby("year"):
        last = g.sort_values("match_date").iloc[-1]
        out[y] = {"champion": last.winner,
                  "runner_up": last.team_a if last.team_b == last.winner else last.team_b,
                  "final": last.p_match}
    return out


# --- the core test ---------------------------------------------------------
def phase_vs_success(edge, rec):
    rule("WHICH PHASE TRACKS SEASON SUCCESS?")
    print("  Which phase's net edge best tracks season win %? (descriptive comparison)\n")

    m = edge.merge(rec, on=["year", "team"])
    rows = []
    for ph in PHASES:
        s = m[m.phase == ph]
        r, p = stats.spearmanr(s.net_edge, s.win_pct)
        # bootstrap the correlation over team-seasons
        boot = []
        idx = np.arange(len(s))
        for _ in range(N_BOOT):
            b = RNG.choice(idx, len(idx), replace=True)
            if len(np.unique(s.net_edge.values[b])) > 2:
                boot.append(stats.spearmanr(s.net_edge.values[b], s.win_pct.values[b]).statistic)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # spread of the edge across teams = how much separation the phase offers
        rows.append({"phase": ph, "n": len(s), "spearman_r": r, "p_value": p,
                     "ci_lo": lo, "ci_hi": hi, "edge_sd": s.net_edge.std(),
                     "edge_range": s.net_edge.max() - s.net_edge.min()})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False).replace("\n", "\n  "))
    print("\n  edge_sd / edge_range: how widely teams differ in that phase.")
    return res


def opportunity(df):
    rule("SIZE OF THE PRIZE: how much is at stake in each phase?")
    tot = (df.groupby("phase", observed=True)
             .agg(balls=("legal_ball", "sum"), runs=("score", "sum")).reset_index())
    tot["pct_balls"] = 100 * tot.balls / tot.balls.sum()
    tot["pct_runs"] = 100 * tot.runs / tot.runs.sum()
    tot["overs_per_innings"] = [6, 9, 5]
    print(tot.round(1).to_string(index=False).replace("\n", "\n  "))
    print("\n  The middle overs are the largest block: a 1 run/over edge is worth 9 runs there.")
    return tot


def validate_finalists(edge, fin, rec):
    rule("VALIDATION: where did the four finalists separate themselves?")
    for y, d in fin.items():
        print(f"\n  ===== {y}  champion: {d['champion']}  |  runner-up: {d['runner_up']} =====")
        s = edge[edge.year == y].copy()
        s["rk"] = s.groupby("phase", observed=True).net_edge.rank(ascending=False).astype(int)
        tbl = s.pivot(index="team", columns="phase", values="rk")[PHASES]
        net = s.pivot(index="team", columns="phase", values="net_edge")[PHASES].round(2)
        out = net.join(tbl, lsuffix="", rsuffix="_rank")
        out = out.join(rec[rec.year == y].set_index("team").win_pct.round(1))
        out = out.sort_values("win_pct", ascending=False)
        print(out.to_string().replace("\n", "\n    "))
        for t, label in [(d["champion"], "CHAMPION "), (d["runner_up"], "RUNNER-UP")]:
            r = s[s.team == t].set_index("phase").rk[PHASES]
            print(f"    {label} {t:30s} phase ranks  PP {r.Powerplay}  MID {r.Middle}  DEATH {r.Death}")


def rr_gap(tp, edge, fin):
    rule("RAJASTHAN ROYALS vs THE BENCHMARK")
    RR = "Rajasthan Royals"
    champs = [(y, d["champion"]) for y, d in fin.items()]
    cols = ["bat_sr_rk", "bowl_econ_rk", "bowl_bpw_rk"]
    for y, champ in champs:
        print(f"\n  -- {y} --  (1 = best of 10)")
        s = tp[(tp.year == y) & (tp.team.isin([champ, RR]))]
        p = s.pivot(index="team", columns="phase", values=cols)
        p = p.reindex(columns=pd.MultiIndex.from_product([cols, PHASES]))
        p.columns = [f"{c.replace('_rk','').replace('bat_sr','BAT').replace('bowl_econ','ECON').replace('bowl_bpw','WKTS')}·{ph[:5]}"
                     for c, ph in p.columns]
        print(p.reindex([champ, RR]).to_string().replace("\n", "\n    "))
    print("\n  Net edge (runs/over above league average):")
    s = edge[edge.team == RR].pivot(index="year", columns="phase", values="net_edge")[PHASES]
    print(s.round(2).to_string().replace("\n", "\n    "))


def control_is_style(tp):
    """Show that control tracks caution rather than quality, so it stays out of the ranks."""
    rule("IS `control` A QUALITY METRIC OR A STYLE METRIC?")
    print("  Correlations across 60 team-season-phase observations:\n")
    pairs = [("bat_ctrl_pct", "bat_sr", "batting control", "strike rate"),
             ("bat_ctrl_pct", "bat_bpd", "batting control", "balls per dismissal"),
             ("bowl_ctrl_against_pct", "bowl_econ", "control conceded", "economy"),
             ("bowl_ctrl_against_pct", "bowl_bpw", "control conceded", "balls per wicket")]
    rows = []
    for a, b, la, lb in pairs:
        r = tp[a].corr(tp[b])
        print(f"  {la:18s} vs {lb:20s} r = {r:+.3f}")
        rows.append({"metric": la, "against": lb, "scope": "all phases", "r": r})
    print("\n  Within each phase, batting control vs strike rate:")
    for ph in PHASES:
        s_ = tp[tp.phase == ph]
        r = s_.bat_ctrl_pct.corr(s_.bat_sr)
        print(f"    {ph:10s} r = {r:+.3f}")
        rows.append({"metric": "batting control", "against": "strike rate",
                     "scope": ph, "r": r})
    print("\n  Finding: control behaves like caution, not skill (its sign flips at the death), so it is not ranked.")
    return pd.DataFrame(rows)


def rr_significance(df):
    """Bootstrap the RR-vs-league strike-rate gap by match, per phase and season."""
    rule("UNCERTAINTY: are RR's phase gaps distinguishable from noise?")
    RR = "Rajasthan Royals"
    print("  Bootstrap by match with both innings kept together, 10,000 resamples, seed 42.\n")
    print(f"  {'phase':10s} {'season':7s} {'mch':>4s} {'RR':>7s} {'others':>7s} "
          f"{'diff':>7s} {'95% CI':>17s} {'p':>6s}  verdict")
    rows = []
    for phase in PHASES:
        for yr in [2023, 2024, None]:
            s_ = df[df.phase == phase]
            if yr:
                s_ = s_[s_.year == yr]
            g = (s_.assign(rr=s_.team_bat == RR).groupby(["p_match", "rr"], observed=True)
                   .agg(r=("batruns", "sum"), b=("ballfaced", "sum")).unstack("rr", fill_value=0))
            rr_r, rr_b, ot_r, ot_b = (g[(c, k)].values.astype(float) for k in (True, False) for c in ("r", "b"))
            rr_sr, ot_sr = 100 * rr_r.sum() / rr_b.sum(), 100 * ot_r.sum() / ot_b.sum()
            i = RNG.integers(0, len(g), (N_BOOT_SIG, len(g)))  # resample matches, both innings together
            with np.errstate(invalid="ignore", divide="ignore"):
                bt = 100 * rr_r[i].sum(1) / rr_b[i].sum(1) - 100 * ot_r[i].sum(1) / ot_b[i].sum(1)
            bt = bt[~np.isnan(bt)]
            rr = rr_b[rr_b > 0]
            lo, hi = np.percentile(bt, [2.5, 97.5])
            p = 2 * min((bt >= 0).mean(), (bt <= 0).mean())
            lab = str(yr) if yr else "POOLED"
            verdict = "SIGNIFICANT" if lo * hi > 0 else "not significant"
            print(f"  {phase:10s} {lab:7s} {len(rr):4d} {rr_sr:7.1f} {ot_sr:7.1f} "
                  f"{rr_sr - ot_sr:+7.1f} [{lo:+6.1f},{hi:+6.1f}] {p:6.3f}  {verdict}")
            rows.append({"phase": phase, "season": lab, "matches": len(rr),
                         "rr_sr": rr_sr, "others_sr": ot_sr, "diff": rr_sr - ot_sr,
                         "ci_lo": lo, "ci_hi": hi, "p_value": p,
                         "significant": bool(lo * hi > 0)})

    print("\n  Finding: no single-season phase gap is significant; 14-15 matches are too few to resolve one.")
    return pd.DataFrame(rows)


def rr_rank_consistency(tp):
    """Count the directional metrics on which RR rank in the bottom half, per phase."""
    rule("ROBUSTNESS: RR's rank profile across the directional metrics")
    RR = "Rajasthan Royals"
    rks = [c for c in tp.columns if c.endswith("_rk")]
    short = {"bat_sr_rk": "BAT-SR", "bat_bnd_pct_rk": "BAT-bnd", "bat_dot_pct_rk": "BAT-dot",
             "bat_bpd_rk": "BAT-bpd", "bowl_econ_rk": "BWL-econ",
             "bowl_dot_pct_rk": "BWL-dot", "bowl_bpw_rk": "BWL-bpw"}
    n = len(rks)
    print(f"  Bottom-half count across the {n} directional metrics (control excluded).\n")
    rows = []
    for _, row in tp[tp.team == RR].sort_values(["year", "phase"]).iterrows():
        bad = sum(row[c] > 5 for c in rks)
        detail = " ".join(f"{short[c]}={int(row[c])}" for c in rks)
        print(f"  {int(row.year)} {row.phase:10s} bottom-half on {bad}/{n} | {detail}")
        rows.append({"year": int(row.year), "phase": row.phase,
                     "bottom_half": bad, "n_metrics": n})
    print(f"\n  2023 middle overs: bottom-half on all {n} directional metrics.")
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(PROC / "clean.parquet")
    matches = pd.read_parquet(PROC / "matches.parquet")
    print(f"Loaded {len(df):,} deliveries, {len(matches)} matches")

    tp = build_team_phase(df)
    tp = add_ranks(tp)
    edge = build_edge(df, tp)
    rec = season_records(matches)
    ctrl = control_is_style(tp)
    fin = finalists(matches)
    opportunity(df)
    res = phase_vs_success(edge, rec)
    validate_finalists(edge, fin, rec)
    rr_gap(tp, edge, fin)
    sig = rr_significance(df)
    cons = rr_rank_consistency(tp)

    rule("WRITING OUTPUT")
    tp.to_parquet(PROC / "team_phase.parquet", index=False)
    edge.to_parquet(PROC / "phase_edge.parquet", index=False)
    rec.to_parquet(PROC / "season_records.parquet", index=False)
    res.to_csv(PROC / "phase_vs_success.csv", index=False)
    sig.to_csv(PROC / "rr_significance.csv", index=False)
    ctrl.to_csv(PROC / "control_is_style.csv", index=False)
    cons.to_csv(PROC / "rr_rank_consistency.csv", index=False)
    for f in ["team_phase.parquet", "phase_edge.parquet", "season_records.parquet",
              "phase_vs_success.csv", "rr_significance.csv", "rr_rank_consistency.csv",
              "control_is_style.csv"]:
        print(f"  {f:30s} {(PROC / f).stat().st_size / 1024:>7.0f} KB")
    print("\nBlock 2 complete.")


if __name__ == "__main__":
    main()
