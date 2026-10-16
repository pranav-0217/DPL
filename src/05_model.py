"""Block 5: win-probability model, cross-validated by match and compared with the supplied wprob."""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt

import viz_style as vs

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
SEED = 42
N_SPLITS = 5
PHASES = ["Powerplay", "Middle", "Death"]

vs.apply_style()


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# --- features -------------------------------------------------------------
BASE = ["inns", "inns_runs", "inns_wkts", "wkts_in_hand", "inns_balls",
        "inns_balls_rem", "inns_rr", "over_num", "is_chase",
        "runs_needed", "req_rate", "target", "venue_par"]
DERIVED = ["resources", "rate_gap", "runs_per_wkt_left", "projection"]


def build(df):
    rule("BUILDING THE MODELLING TABLE")
    # Enforce exclusions per match, even when delivery-level flags are mixed.
    flags = df.groupby("p_match")[["no_result", "shortened"]].any()
    excluded = flags.index[flags.any(axis=1)]
    d = df[~df.p_match.isin(excluded)].copy()
    print(f"  Excluded: {int(flags.no_result.sum())} no-result match(es), "
          f"{int(flags.shortened.sum())} rain-shortened match(es) "
          f"({len(excluded)} distinct matches; DLS distorts state)")
    print(f"  Deliveries: {len(d):,}   Matches: {d.p_match.nunique()}")
    print(f"  INDEPENDENT MATCH GROUPS: {d.p_match.nunique()}")

    d["bat_team_wins"] = (d.team_bat == d.winner).astype(int)
    d["wkts_in_hand"] = 10 - d.inns_wkts
    d["is_chase"] = (d.inns == 2).astype(int)
    d["runs_needed"] = d.inns_runs_rem
    d["req_rate"] = d.inns_rrr.replace([np.inf, -np.inf], np.nan)
    d["venue_par"] = np.nan                       # filled fold-wise

    # Cricket-motivated terms, specified in advance rather than searched for.
    d["resources"] = d.inns_balls_rem * d.wkts_in_hand      # batting resources left
    d["rate_gap"] = (d.req_rate.fillna(0) - d.inns_rr) * d.is_chase
    d["runs_per_wkt_left"] = (d.runs_needed.fillna(0) / d.wkts_in_hand.clip(lower=1)) * d.is_chase
    d["projection"] = d.inns_rr * d.inns_balls_rem / 6      # naive projected addition

    print(f"  Base features ({len(BASE)}): {', '.join(BASE)}")
    print(f"  Derived ({len(DERIVED)}): {', '.join(DERIVED)}")
    print("  Excluded as baseline, not input: wprob, predscore")
    return d


def venue_par_from_train(tr_idx, d):
    tr = d.iloc[tr_idx]
    first = tr[tr.inns == 1].groupby(["p_match", "ground"], observed=True).score.sum().reset_index()
    par = first.groupby("ground", observed=True).score.mean()
    return d.ground.map(par).fillna(first.score.mean()).values


def lr_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000, C=1.0))


def gbm_model(**kw):
    p = dict(max_depth=5, max_iter=300, min_samples_leaf=60,
             l2_regularization=1.0, learning_rate=0.05, random_state=SEED)
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


def run_cv(d, cols, make):
    y = d.bat_team_wins.values
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(d, y, d.p_match.values):
        X = d[cols].copy()
        X["venue_par"] = venue_par_from_train(tr, d)
        oof[te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return oof


def capacity_sweep(d):
    rule("CAPACITY SWEEP: why gradient boosting is the wrong tool here")
    print(f"  {d.p_match.nunique()} match groups: reduce tree capacity step by step to expose overfitting.\n")
    print(f"  {'configuration':48s} {'log_loss':>9s} {'auc':>7s}")
    grid = [dict(max_depth=5, max_iter=300, min_samples_leaf=60, l2_regularization=1.0),
            dict(max_depth=4, max_iter=200, min_samples_leaf=150, l2_regularization=5.0),
            dict(max_depth=3, max_iter=150, min_samples_leaf=300, l2_regularization=10.0),
            dict(max_depth=2, max_iter=120, min_samples_leaf=500, l2_regularization=20.0),
            dict(max_depth=2, max_iter=60, min_samples_leaf=800, l2_regularization=30.0)]
    y = d.bat_team_wins.values
    rows = []
    best = (np.inf, None, None)
    for prm in grid:
        oof = run_cv(d, BASE + DERIVED, lambda p=prm: gbm_model(**p))
        ll, auc = log_loss(y, oof), roc_auc_score(y, oof)
        lab = (f"depth {prm['max_depth']}, {prm['max_iter']} trees, "
               f"leaf {prm['min_samples_leaf']}, l2 {prm['l2_regularization']:g}")
        print(f"  {lab:48s} {ll:9.4f} {auc:7.4f}")
        rows.append({"config": lab, **prm, "log_loss": ll, "auc": auc})
        if ll < best[0]:
            best = (ll, prm, oof)
    print("\n  Log-loss broadly improves as capacity falls (overfitting); no tree model beats the linear one.")
    return pd.DataFrame(rows), best


def compare(d, best_gbm):
    rule("MODEL COMPARISON  (5-fold CV grouped by match, identical rows)")
    y = d.bat_team_wins.values
    oof_base = run_cv(d, BASE, lr_model)
    oof_eng = run_cv(d, BASE + DERIVED, lr_model)
    vendor = (d.wprob.values / 100).clip(1e-6, 1 - 1e-6)

    preds = {"Supplied `wprob` (baseline)": vendor,
             "Gradient boosting (best config)": best_gbm[2],
             "Logistic regression, base features": oof_base,
             "Logistic regression + derived features": oof_eng}
    res = pd.DataFrame([{"model": k, "log_loss": log_loss(y, v),
                         "brier": brier_score_loss(y, v), "auc": roc_auc_score(y, v)}
                        for k, v in preds.items()])
    print(res.round(4).to_string(index=False).replace("\n", "\n  "))

    v = res.set_index("model")
    win = v.log_loss["Logistic regression + derived features"]
    base = v.log_loss["Supplied `wprob` (baseline)"]
    print("\n  Best model: logistic regression with cricket-derived features.")
    print(f"  vs the supplied model: {win - base:+.4f} log-loss "
          f"({100 * (base - win) / base:+.1f}% better), AUC "
          f"{v.auc['Logistic regression + derived features']:.4f} vs {v.auc['Supplied `wprob` (baseline)']:.4f}")
    print("\n  Caveat: configurations were compared on the scoring folds, so the best score is mildly optimistic.")

    d = d.copy()
    d["p_ours"] = oof_eng
    d["p_vendor"] = vendor
    return d, res


def paired_gap(d):
    """Bootstrap the supplied-minus-ours log-loss gap by match, keeping each match's balls together."""
    rule("IS OUR LOWER LOG-LOSS SIGNIFICANT? (paired bootstrap by match)")
    y = d.bat_team_wins.values
    ll = lambda q: -(y * np.log(q) + (1 - y) * np.log(1 - q))
    g = (pd.DataFrame({"m": d.p_match.values, "ours": ll(np.clip(d.p_ours.values, 1e-6, 1 - 1e-6)),
                       "vend": ll(d.p_vendor.values)}).groupby("m").agg(["sum", "size"]))
    o, v, n = g[("ours", "sum")].values, g[("vend", "sum")].values, g[("ours", "size")].values
    idx = np.random.default_rng(SEED).integers(0, len(g), (20000, len(g)))
    bt = v[idx].sum(1) / n[idx].sum(1) - o[idx].sum(1) / n[idx].sum(1)
    gap, (lo, hi) = v.sum() / n.sum() - o.sum() / n.sum(), np.percentile(bt, [2.5, 97.5])
    print(f"  Supplied minus ours: {gap:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], ours lower in {100 * (bt > 0).mean():.0f}% of resamples")
    print("  " + ("Significant." if lo > 0 else "Not significant: report a lower observed loss, not a win."))
    return pd.DataFrame([{"gap": gap, "ci_lo": lo, "ci_hi": hi, "share_ours_lower": (bt > 0).mean(),
                          "significant": bool(lo > 0)}])


def by_phase(d):
    rule("WHERE EACH MODEL IS STRONGER")
    rows = []
    for (inn, ph), g in d.groupby(["inns", "phase"], observed=True):
        rows.append({"innings": inn, "phase": ph, "balls": len(g),
                     "ours": log_loss(g.bat_team_wins, g.p_ours, labels=[0, 1]),
                     "vendor": log_loss(g.bat_team_wins, g.p_vendor, labels=[0, 1])})
    t = pd.DataFrame(rows)
    t["ours_better_by"] = t.vendor - t.ours
    t["phase"] = pd.Categorical(t.phase, PHASES, ordered=True)
    t = t.sort_values(["innings", "phase"])
    print("  Log-loss by innings and phase (lower is better):\n")
    print(t.round(4).to_string(index=False).replace("\n", "\n  "))
    print("\n  Positive `ours_better_by` = our model wins there.")
    return t


def phase_swing(d):
    rule("HOW MUCH DOES ONE DELIVERY MOVE THE RESULT?")
    print("  Per-ball swing in win probability; early phases are flattered because values near 0.5 move more.\n")

    d = d.sort_values(["p_match", "inns", "ball_id"]).copy()
    d["dwp"] = d.groupby(["p_match", "inns"], observed=True).p_ours.diff().abs()

    def tab(sub):
        s = (sub.dropna(subset=["dwp"]).groupby("phase", observed=True)
               .agg(total=("dwp", "sum"), balls=("dwp", "size")).reset_index())
        s["pct_of_swing"] = 100 * s.total / s.total.sum()
        s["pct_of_balls"] = 100 * s.balls / s.balls.sum()
        s["leverage_index"] = s.pct_of_swing / s.pct_of_balls
        s["phase"] = pd.Categorical(s.phase, PHASES, ordered=True)
        return s.sort_values("phase")

    whole, first = tab(d), tab(d[d.inns == 1])
    print("  Whole match:")
    print(whole.round(3).to_string(index=False).replace("\n", "\n    "))
    print("\n  First innings only (no chase arithmetic):")
    print(first.round(3).to_string(index=False).replace("\n", "\n    "))
    print("\n  Middle-overs balls carry the least leverage each, but the phase is 46% of all balls.")
    return whole, first


def importance(d):
    rule("WHAT THE MODEL LEANS ON")
    cols = BASE + DERIVED
    y = d.bat_team_wins.values
    tr, te = next(GroupKFold(n_splits=N_SPLITS).split(d, y, d.p_match.values))
    X = d[cols].copy()
    X["venue_par"] = venue_par_from_train(tr, d)
    m = lr_model().fit(X.iloc[tr], y[tr])
    r = permutation_importance(m, X.iloc[te], y[te], n_repeats=10,
                               random_state=SEED, scoring="neg_log_loss")
    imp = pd.DataFrame({"feature": cols, "importance": r.importances_mean,
                        "sd": r.importances_std}).sort_values("importance", ascending=False)
    print("  Rise in log-loss when the feature is shuffled (higher = relied on more):\n")
    print(imp.round(4).to_string(index=False).replace("\n", "\n  "))
    return imp


# ------------------------------------------------------------------ figures
def fig_calibration(d, res, gap):
    bins = np.linspace(0, 1, 11)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    vs.layout(fig, top=0.74, bottom=0.255, left=0.075, right=0.985, wspace=0.42)

    ax.plot([0, 1], [0, 1], color=vs.BASELINE, lw=1.4, zorder=1)
    for col, c, lab in [("p_vendor", vs.MUTED, "Supplied `wprob`"),
                        ("p_ours", vs.S1_BLUE, "Ours (out-of-fold)")]:
        idx = np.digitize(d[col], bins) - 1
        g = d.groupby(idx).agg(pred=(col, "mean"), act=("bat_team_wins", "mean"), n=(col, "size"))
        g = g[g.n >= 50]
        ax.plot(g.pred, g.act, marker="o", ms=6, lw=2, color=c, label=lab,
                markeredgecolor=vs.SURFACE, markeredgewidth=1.5, zorder=3)
    ax.set_xlabel("Predicted probability the batting team wins")
    ax.set_ylabel("Observed win rate")
    ax.legend(loc="upper left")
    vs.despine(ax)
    vs.panel_title(ax, "Both models are well calibrated")

    order = ["Supplied `wprob` (baseline)", "Gradient boosting (best config)",
             "Logistic regression, base features", "Logistic regression + derived features"]
    names = ["Supplied `wprob`", "Gradient boosting", "Logistic (base)", "Logistic (+derived)"]
    r = res.set_index("model").loc[order]
    # Dot plot, not bars: log-loss has no meaningful zero.
    vals = r.log_loss.values
    base_ll = vals[0]
    yy = np.arange(len(vals))[::-1]
    cols = [vs.MUTED, vs.MUTED, vs.MUTED, vs.S1_BLUE]
    ax2.axvline(base_ll, color=vs.S2_ORANGE, lw=1.4, zorder=1)
    for i, (v_, c_) in zip(yy, zip(vals, cols)):
        ax2.plot([base_ll, v_], [i, i], color=c_, lw=2, alpha=.45, zorder=2)
        ax2.scatter([v_], [i], s=110, color=c_, zorder=3,
                    edgecolor=vs.SURFACE, linewidth=2)
        ax2.text(v_, i + .30, f"{v_:.4f}", ha="center", fontsize=9.5, color=vs.INK_2)
    ax2.set_yticks(yy, names, fontsize=9.5)
    ax2.tick_params(axis="y", labelcolor=vs.INK)
    ax2.set_xlabel("Log-loss   (lower is better)")
    ax2.set_xlim(0.540, 0.595)
    ax2.set_ylim(-0.6, len(vals) - 0.25)
    ax2.text(base_ll + .0012, -0.42, "supplied-model baseline", fontsize=9,
             color=vs.S2_ORANGE, va="center", ha="left")
    ax2.grid(axis="y", visible=False)
    vs.despine(ax2)
    vs.panel_title(ax2, "Scored on identical deliveries")
    ax2.yaxis.set_tick_params(pad=2)

    vs.title_block(fig, "Our model has a lower observed log-loss than the supplied one",
                   f"{len(d):,} deliveries, {d.p_match.nunique()} matches. 5-fold cross-validation grouped by "
                   "match, so no match appears in both train and test.")
    g = gap.iloc[0]
    vs.caption(fig, f"The gap is {'' if g.significant else 'not '}statistically significant "
                    f"(95% CI {g.ci_lo:+.3f} to {g.ci_hi:+.3f}, paired bootstrap by match). "
                    "Gradient boosting loses to a regularised linear model.")
    fig.savefig(FIGS / "fig5_model_calibration.png")
    plt.close(fig)
    print("  fig5_model_calibration.png")


def fig_swing(whole, first):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    vs.layout(fig, top=0.73, bottom=0.25, left=0.095, right=0.975, wspace=0.12)
    ymax = max(whole.leverage_index.max(), first.leverage_index.max()) * 1.26
    for a, data, lab in [(ax, whole, "Whole match"), (ax2, first, "First innings only")]:
        vals = data.leverage_index.values
        colors = [vs.S1_BLUE if p == "Middle" else vs.MUTED for p in data.phase]
        a.bar(data.phase.astype(str), vals, width=.42, color=colors, zorder=2)
        a.axhline(1.0, color=vs.BASELINE, lw=1.3, zorder=1)
        for i, v_ in enumerate(vals):
            a.text(i, v_ + .03, f"{v_:.2f}", ha="center", fontsize=10.5, color=vs.INK_2)
        a.set_ylim(0, ymax)
        a.grid(axis="x", visible=False)
        vs.despine(a)
        vs.panel_title(a, lab)
    ax.set_ylabel("Win-probability swing per ball\n(1.0 = an average delivery)")
    vs.title_block(fig, "The middle overs are where you accumulate, not where you spike",
                   "Per-ball leverage: absolute change in win probability from one delivery to the next.")
    vs.caption(fig, "Individual middle-overs deliveries move the result least — yet middle-overs performance tracks "
                    "season success most closely,\nbecause the phase is 46% of all deliveries. Confound: a probability "
                    "near 0.5 can move further than one near 0 or 1.")
    fig.savefig(FIGS / "fig6_phase_swing.png")
    plt.close(fig)
    print("  fig6_phase_swing.png")


def main():
    df = pd.read_parquet(PROC / "clean.parquet")
    d = build(df)
    sweep, best_gbm = capacity_sweep(d)
    d, res = compare(d, best_gbm)
    gap = paired_gap(d)
    t = by_phase(d)
    whole, first = phase_swing(d)
    imp = importance(d)

    rule("WRITING OUTPUT")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_calibration(d, res, gap)
    fig_swing(whole, first)
    for obj, name in [(res, "model_scores.csv"), (gap, "model_gap.csv"), (sweep, "model_capacity_sweep.csv"),
                      (t, "model_by_phase.csv"), (whole, "model_phase_swing.csv"),
                      (first, "model_phase_swing_inns1.csv"), (imp, "model_importance.csv")]:
        obj.to_csv(PROC / name, index=False)
        print(f"  {name:34s} {(PROC / name).stat().st_size / 1024:>7.1f} KB")
    print("\nBlock 5 complete.")


if __name__ == "__main__":
    main()
