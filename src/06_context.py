"""Block 6: venue, opposition, line/length and shot-intent context for RR's middle overs."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
RNG = np.random.default_rng(42)
N_BOOT = 10000
RR = "Rajasthan Royals"
MIN_BALLS = 25            # suppression threshold for venue / opposition splits
MIN_BALLS_CELL = 50       # hide pitch-map cells with fewer than 50 balls

# Merge each wide line band into its neighbour so every cell has enough balls.
LINE_GROUP = {"WIDE_DOWN_LEG": "Leg side", "DOWN_LEG": "Leg side",
              "ON_THE_STUMPS": "At the stumps",
              "OUTSIDE_OFFSTUMP": "Outside off", "WIDE_OUTSIDE_OFFSTUMP": "Outside off"}
LINES = ["Leg side", "At the stumps", "Outside off"]
LENGTHS = ["YORKER", "FULL_TOSS", "FULL", "GOOD_LENGTH", "SHORT_OF_A_GOOD_LENGTH", "SHORT"]

# Attacking shots take risk for boundaries; rotational shots keep the strike moving.
ATTACKING = {"SLOG_SHOT", "SLOG_SWEEP", "PULL", "HOOK", "UPPER_CUT", "RAMP",
             "REVERSE_SCOOP", "REVERSE_PULL", "SWEEP_SHOT", "REVERSE_SWEEP"}
ORTHODOX = {"COVER_DRIVE", "ON_DRIVE", "STRAIGHT_DRIVE", "SQUARE_DRIVE",
            "FLICK", "CUT_SHOT", "LEG_GLANCE"}
ROTATIONAL = {"DEFENDED", "LEFT_ALONE", "PUSH", "DAB", "STEERED",
              "LATE_CUT", "PADDLE_SWEEP", "PADDLE_AWAY"}

# Wagon zones are ground-fixed, so flip left-handers to compare leg side with leg side.
ZONE_FLIP = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 0: 0}
ZONE_NAME = {1: "Fine leg", 2: "Square leg", 3: "Midwicket", 4: "Long on",
             5: "Long off", 6: "Cover", 7: "Point", 8: "Third man"}
SHORT = {"Chennai Super Kings": "CSK", "Delhi Capitals": "DC", "Gujarat Titans": "GT",
         "Kolkata Knight Riders": "KKR", "Lucknow Super Giants": "LSG",
         "Mumbai Indians": "MI", "Punjab Kings": "PBKS", "Rajasthan Royals": "RR",
         "Royal Challengers Bengaluru": "RCB", "Sunrisers Hyderabad": "SRH"}

vs.apply_style()


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def boot_sr_diff(a, b):
    """Bootstrap a strike-rate difference by match."""
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    obs = 100 * (a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum())
    i = RNG.integers(0, len(a), (N_BOOT, len(a)))
    j = RNG.integers(0, len(b), (N_BOOT, len(b)))
    bt = 100 * (a[i, 0].sum(1) / a[i, 1].sum(1) - b[j, 0].sum(1) / b[j, 1].sum(1))
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return obs, lo, hi


def pm(d):
    g = d.groupby("p_match", observed=True).agg(r=("batruns", "sum"), b=("ballfaced", "sum"))
    return g[g.b > 0][["r", "b"]].values.astype(float)


# --------------------------------------------------------------------- 1
def venue(df):
    rule("1  VENUE: does the middle-overs problem travel?")
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1)]
    rr = mid[mid.team_bat == RR]
    print(f"  RR middle-overs balls by ground (minimum {MIN_BALLS} shown):\n")
    print(f"  {'ground':52s} {'mch':>4s} {'balls':>6s} {'RR SR':>7s} {'lg SR here':>11s} {'gap':>7s}")
    rows = []
    for gr, s in rr.groupby("ground", observed=True):
        if s.ballfaced.sum() < MIN_BALLS:
            continue
        lg = mid[(mid.ground == gr) & (mid.team_bat != RR)]
        rr_sr = 100 * s.batruns.sum() / s.ballfaced.sum()
        lg_sr = 100 * lg.batruns.sum() / lg.ballfaced.sum() if len(lg) else np.nan
        rows.append({"ground": gr, "matches": s.p_match.nunique(),
                     "balls": int(s.ballfaced.sum()), "rr_sr": rr_sr,
                     "league_sr_here": lg_sr, "gap": rr_sr - lg_sr})
        print(f"  {gr[:52]:52s} {s.p_match.nunique():4d} {int(s.ballfaced.sum()):6d} "
              f"{rr_sr:7.1f} {lg_sr:11.1f} {rr_sr - lg_sr:+7.1f}")

    jai = "Sawai Mansingh Stadium, Jaipur"
    home = rr[rr.ground == jai]
    away = rr[rr.ground != jai]
    o, lo, hi = boot_sr_diff(pm(home), pm(away))
    print("\n  Home (Jaipur) vs away, RR middle overs:")
    print(f"    home {100*home.batruns.sum()/home.ballfaced.sum():.1f} "
          f"({home.p_match.nunique()} matches)   "
          f"away {100*away.batruns.sum()/away.ballfaced.sum():.1f} "
          f"({away.p_match.nunique()} matches)   "
          f"diff {o:+.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]")
    print("    -> " + ("home/away difference is significant"
                       if lo * hi > 0 else "no significant home/away difference"))
    print("\n  Each gap compares RR with other teams at the same ground and phase.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- 2
def opposition(df):
    rule("2  OPPOSITION: who exploits the middle overs, and with what?")
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1)]
    rr = mid[mid.team_bat == RR]
    print(f"  RR middle-overs batting vs each attack (minimum {MIN_BALLS} balls):\n")
    print(f"  {'opponent':30s} {'mch':>4s} {'balls':>6s} {'RR SR':>7s} {'their econ vs others':>21s}")
    rows = []
    for op, s in rr.groupby("team_bowl", observed=True):
        if s.ballfaced.sum() < MIN_BALLS:
            continue
        others = mid[(mid.team_bowl == op) & (mid.team_bat != RR)]
        sr = 100 * s.batruns.sum() / s.ballfaced.sum()
        their = 100 * others.batruns.sum() / others.ballfaced.sum() if len(others) else np.nan
        rows.append({"opponent": op, "matches": s.p_match.nunique(),
                     "balls": int(s.ballfaced.sum()), "rr_sr": sr, "sr_allowed_to_others": their})
        print(f"  {op:30s} {s.p_match.nunique():4d} {int(s.ballfaced.sum()):6d} {sr:7.1f} {their:21.1f}")
    t = pd.DataFrame(rows)
    print(f"\n  RR were held below 120 by: "
          f"{', '.join(t[t.rr_sr < 120].opponent.str.split().str[0]) or 'none'}")
    print("  Caution: 2-4 matches per opponent; scouting notes, not conclusions.")

    print("\n  By bowling style faced in the middle overs (pooled, both seasons):")
    print(f"  {'style':10s} {'RR balls':>9s} {'RR SR':>7s} {'league SR':>10s} {'gap':>7s}")
    srows = []
    for st, s in mid[mid.team_bat == RR].groupby("bowl_style", observed=True):
        if s.ballfaced.sum() < 60:
            continue
        lg = mid[(mid.bowl_style == st) & (mid.team_bat != RR)]
        sr = 100 * s.batruns.sum() / s.ballfaced.sum()
        lsr = 100 * lg.batruns.sum() / lg.ballfaced.sum()
        srows.append({"bowl_style": st, "balls": int(s.ballfaced.sum()),
                      "rr_sr": sr, "league_sr": lsr, "gap": sr - lsr})
        print(f"  {st:10s} {int(s.ballfaced.sum()):9d} {sr:7.1f} {lsr:10.1f} {sr - lsr:+7.1f}")
    return t, pd.DataFrame(srows)


# --------------------------------------------------------------------- 3
def pitch_map(df):
    rule("3  PITCH MAP: which deliveries stall RR in the middle overs")
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1)].copy()
    mid["line_grp"] = mid.line.map(LINE_GROUP)
    rr, lg = mid[mid.team_bat == RR], mid[mid.team_bat != RR]

    def grid(d, val):
        t = d.pivot_table(index="length", columns="line_grp", values=val,
                          aggfunc="sum", observed=True)
        n = d.pivot_table(index="length", columns="line_grp", values="ballfaced",
                          aggfunc="sum", observed=True)
        return t.reindex(index=LENGTHS, columns=LINES), n.reindex(index=LENGTHS, columns=LINES)

    r_runs, r_n = grid(rr, "batruns")
    l_runs, l_n = grid(lg, "batruns")
    rr_sr = 100 * r_runs / r_n
    lg_sr = 100 * l_runs / l_n
    diff = (rr_sr - lg_sr).where(r_n >= MIN_BALLS_CELL)
    keep = diff.notna().any(axis=1)
    diff, r_n, rr_sr, lg_sr = diff[keep], r_n[keep], rr_sr[keep], lg_sr[keep]

    print(f"  RR strike rate minus league by line and length (cells under {MIN_BALLS_CELL} balls hidden):\n")
    pr = diff.round(0)
    pr.index = [i.replace("_", " ").title() for i in pr.index]
    pr.columns = [c.replace("_", " ").title() for c in pr.columns]
    print(pr.to_string(na_rep="  .").replace("\n", "\n  "))

    flat = diff.stack().dropna().sort_values()
    print("\n  Worst three:")
    for (le, li), v in flat.head(3).items():
        print(f"    {le:24s} {li:22s} {v:+6.1f}  "
              f"(RR {rr_sr.loc[le, li]:.0f} vs league {lg_sr.loc[le, li]:.0f}, "
              f"{int(r_n.loc[le, li])} balls)")
    print("  Best three:")
    for (le, li), v in flat.tail(3).items():
        print(f"    {le:24s} {li:22s} {v:+6.1f}  "
              f"(RR {rr_sr.loc[le, li]:.0f} vs league {lg_sr.loc[le, li]:.0f}, "
              f"{int(r_n.loc[le, li])} balls)")

    out = diff.stack().rename("sr_diff").reset_index()
    out["rr_balls"] = [r_n.loc[a, b] for a, b in zip(out.length, out.line_grp)]
    return diff, r_n, out


# --------------------------------------------------------------------- 4
def shots(df):
    rule("4  SHOT SELECTION: choosing safety, or forced into it?")
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1) & df.shot.notna()]

    def grp(s):
        return ("Attacking" if s in ATTACKING else
                "Orthodox scoring" if s in ORTHODOX else
                "Rotational / defensive")

    mid = mid.assign(intent=mid.shot.map(grp))
    print("  Share of middle-overs deliveries by shot intent, and what each returns:\n")
    print(f"  {'team':6s} {'Attacking':>10s} {'Orthodox':>10s} {'Rotational':>11s}   "
          f"{'SR|attack':>10s} {'SR|rot':>8s}")
    rows = []
    for t, s in mid.groupby("team_bat", observed=True):
        sh = s.intent.value_counts(normalize=True) * 100
        a = s[s.intent == "Attacking"]
        r_ = s[s.intent == "Rotational / defensive"]
        rows.append({"team": t, "attacking_pct": sh.get("Attacking", 0),
                     "orthodox_pct": sh.get("Orthodox scoring", 0),
                     "rotational_pct": sh.get("Rotational / defensive", 0),
                     "sr_attacking": 100 * a.batruns.sum() / max(len(a), 1),
                     "sr_rotational": 100 * r_.batruns.sum() / max(len(r_), 1)})
    t = pd.DataFrame(rows).sort_values("attacking_pct", ascending=False)
    for _, r in t.iterrows():
        mark = "  <-- RR" if r.team == RR else ""
        print(f"  {SHORT[r.team]:6s} {r.attacking_pct:9.1f}% {r.orthodox_pct:9.1f}% "
              f"{r.rotational_pct:10.1f}%   {r.sr_attacking:10.1f} {r.sr_rotational:8.1f}{mark}")
    rr_row = t[t.team == RR].iloc[0]
    rank = int((t.attacking_pct > rr_row.attacking_pct).sum() + 1)
    print(f"\n  RR attacking-shot share: {rr_row.attacking_pct:.1f}% -- rank {rank} of 10.")
    a_rank = int((t.sr_attacking > rr_row.sr_attacking).sum() + 1)
    print(f"  Yet when RR DO attack they score at {rr_row.sr_attacking:.0f} "
          f"-- rank {a_rank} of 10 in the league.")
    print("  Least willing to attack, best returns when they do.")
    return t


# --------------------------------------------------------------------- 5
def wagon(df):
    rule("5  SCORING AREAS: where RR's middle-overs runs go")
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1) & (df.wagonzone > 0)].copy()
    mid["zone"] = np.where(mid.bat_hand == "LHB",
                           mid.wagonzone.map(ZONE_FLIP), mid.wagonzone)
    print("  Zones flipped for left-handers so leg side means leg side for both.\n")
    rr, lg = mid[mid.team_bat == RR], mid[mid.team_bat != RR]
    print(f"  {'zone':13s} {'RR share':>9s} {'lg share':>9s} {'RR runs/ball':>13s} {'lg runs/ball':>13s}")
    rows = []
    for z in range(1, 9):
        a, b = rr[rr.zone == z], lg[lg.zone == z]
        rows.append({"zone": z, "name": ZONE_NAME[z],
                     "rr_share": 100 * len(a) / len(rr), "lg_share": 100 * len(b) / len(lg),
                     "rr_rpb": a.batruns.mean(), "lg_rpb": b.batruns.mean()})
        print(f"  {ZONE_NAME[z]:13s} {100*len(a)/len(rr):8.1f}% {100*len(b)/len(lg):8.1f}% "
              f"{a.batruns.mean():13.2f} {b.batruns.mean():13.2f}")
    t = pd.DataFrame(rows)
    t["share_gap"] = t.rr_share - t.lg_share
    worst = t.loc[t.share_gap.idxmin()]
    print(f"\n  Biggest under-use: {worst['name']} "
          f"({worst.rr_share:.1f}% vs league {worst.lg_share:.1f}%)")
    return t


# ------------------------------------------------------------------ figures
def fig_pitch(diff, n):
    n = n.loc[diff.index]
    cmap = LinearSegmentedColormap.from_list("d", [vs.DIV_BAD, vs.DIV_MID, vs.DIV_GOOD])
    lim = np.nanmax(np.abs(diff.values))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    vs.layout(fig, top=0.70, bottom=0.21, left=0.215, right=0.80)
    ax.imshow(diff.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", fontsize=13, color=vs.MUTED)
            else:
                c = "#ffffff" if abs(v) > lim * .55 else vs.INK
                ax.text(j, i - .13, f"{v:+.0f}", ha="center", va="center",
                        fontsize=13, color=c, fontweight="bold")
                ax.text(j, i + .22, f"{int(n.values[i, j])} balls", ha="center",
                        va="center", fontsize=8.5, color=c, alpha=.8)
    ax.set_xticks(range(len(LINES)), [c.replace("_", "\n").title() for c in LINES], fontsize=9)
    ax.set_yticks(range(len(diff.index)), [r.replace("_", " ").title() for r in diff.index], fontsize=10)
    ax.tick_params(axis="y", labelcolor=vs.INK)
    ax.set_xticks(np.arange(-.5, len(LINES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(diff.index), 1), minor=True)
    ax.grid(which="minor", color=vs.SURFACE, linewidth=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=.03, pad=.16)
    cb.set_label("RR strike rate minus league", fontsize=9)
    cb.outline.set_visible(False)
    vs.title_block(fig, "Where Rajasthan Royals stall in the middle overs",
                   "Strike rate against the rest of the league, by line and length, overs 7–15, both seasons.\n"
                   f"Cells with fewer than {MIN_BALLS_CELL} RR deliveries are suppressed; lengths with no usable cell are dropped.")
    vs.caption(fig, "Red = RR score slower than the league off that delivery. "
                    "This is the mechanism behind the control finding.")
    fig.savefig(FIGS / "fig7_pitch_map.png")
    plt.close(fig)
    print("  fig7_pitch_map.png")


def fig_shot_venue(sh, ven):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0),
                                  gridspec_kw={"width_ratios": [1, 1.15]})
    vs.layout(fig, top=0.74, bottom=0.26, left=0.075, right=0.975, wspace=0.50)

    s = sh.sort_values("attacking_pct").copy()
    s["T"] = s.team.map(SHORT)
    y = np.arange(len(s))
    cols = [vs.S1_BLUE if t == "RR" else vs.MUTED for t in s["T"]]
    ax.barh(y, s.attacking_pct, height=.52, color=cols, zorder=2)
    for i, v in zip(y, s.attacking_pct):
        ax.text(v + .25, i, f"{v:.1f}%", va="center", fontsize=9.5, color=vs.INK_2)
    ax.set_yticks(y, s["T"], fontsize=10)
    ax.tick_params(axis="y", labelcolor=vs.INK)
    ax.set_xlabel("Share of middle-overs balls played with an attacking shot")
    ax.set_xlim(0, s.attacking_pct.max() * 1.22)
    ax.grid(axis="y", visible=False)
    vs.despine(ax)
    vs.panel_title(ax, "Shot intent, overs 7–15")

    # Show only grounds where RR played two or more matches.
    v = ven[ven.matches >= 2].sort_values("gap").copy()
    v["label"] = v.ground.str.split(",").str[-1].str.strip()
    v.loc[v.label.str.contains("Jaipur"), "label"] = "Jaipur (home)"
    yy = np.arange(len(v))
    cols2 = [vs.DIV_BAD if g < 0 else vs.DIV_GOOD for g in v.gap]
    ax2.barh(yy, v.gap, height=.48, color=cols2, zorder=2)
    for i, (g, b) in zip(yy, zip(v.gap, v.balls)):
        ax2.text(g + (1.5 if g >= 0 else -1.5), i, f"{g:+.0f}",
                 va="center", ha="left" if g >= 0 else "right",
                 fontsize=10, color=vs.INK_2, fontweight="bold")
        ax2.text(v.gap.max() + 14, i, f"{int(b)} balls", va="center",
                 fontsize=9, color=vs.MUTED)
    ax2.axvline(0, color=vs.INK, lw=1.2, zorder=3)
    ax2.set_yticks(yy, v.label, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=vs.INK)
    ax2.set_xlabel("RR middle-overs strike rate minus the league's, at that ground")
    ax2.set_xlim(v.gap.min() - 10, v.gap.max() + 34)
    ax2.grid(axis="y", visible=False)
    vs.despine(ax2)
    vs.panel_title(ax2, "No venue rescues it — not even home")

    vs.title_block(fig, "RR attack less often than any side in the middle overs",
                   "Both seasons pooled. Attacking shots: sweeps, pulls, hooks, slogs, ramps, upper cuts and reverses.")
    vs.caption(fig, "Venue panel shows only grounds where RR played two or more matches. At Jaipur — their own ground and by far "
                    "the largest sample —\nRR score slightly BELOW what visiting teams manage there in the same overs.")
    fig.savefig(FIGS / "fig8_shot_and_venue.png")
    plt.close(fig)
    print("  fig8_shot_and_venue.png")


def main():
    df = pd.read_parquet(PROC / "clean.parquet")
    print(f"Loaded {len(df):,} deliveries")
    ven = venue(df)
    opp, styles = opposition(df)
    diff, n, flat = pitch_map(df)
    sh = shots(df)
    wg = wagon(df)

    rule("WRITING OUTPUT")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_pitch(diff, n)
    fig_shot_venue(sh, ven)
    for obj, name in [(ven, "context_venue.csv"), (opp, "context_opposition.csv"),
                      (styles, "context_bowl_style.csv"), (flat, "context_pitch_map.csv"),
                      (sh, "context_shot_intent.csv"), (wg, "context_wagon.csv")]:
        obj.to_csv(PROC / name, index=False)
        print(f"  {name:30s} {(PROC / name).stat().st_size / 1024:>7.1f} KB")
    print("\nBlock 6 complete.")


if __name__ == "__main__":
    main()
