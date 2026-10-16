"""Block 3: the core analysis figures (fig1-fig4) in reports/figures."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
PHASES = ["Powerplay", "Middle", "Death"]
RR = "Rajasthan Royals"

vs.apply_style()
FIGS.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- figure 1
def fig1():
    d = pd.read_csv(PROC / "phase_vs_success.csv")
    d["phase"] = pd.Categorical(d.phase, PHASES, ordered=True)
    d = d.sort_values("phase")
    y = np.arange(len(d))[::-1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    vs.layout(fig, top=0.72, bottom=0.26, left=0.095, right=0.975, wspace=0.30)

    for i, (_, r) in zip(y, d.iterrows()):
        lead = r.phase == "Middle"
        col = vs.S1_BLUE if lead else vs.MUTED
        axL.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2.2 if lead else 1.6,
                 solid_capstyle="round", alpha=1 if lead else .5, zorder=2)
        axL.scatter([r.spearman_r], [i], s=120 if lead else 70, color=col,
                    zorder=3, edgecolor=vs.SURFACE, linewidth=2)
        txt = f"p = {r.p_value:.3f}" + ("" if r.p_value < .05 else "   not significant")
        axL.text(r.ci_hi + .05, i, txt, va="center", fontsize=9,
                 color=vs.INK if lead else vs.MUTED,
                 fontweight="bold" if lead else "normal")
    axL.axvline(0, color=vs.BASELINE, lw=1, zorder=1)
    axL.set_yticks(y, d.phase.astype(str))
    axL.tick_params(axis="y", labelsize=10.5, labelcolor=vs.INK)
    axL.set_xlabel("Spearman correlation with season win %   (95% CI)")
    axL.set_xlim(-0.35, 1.45)
    axL.grid(axis="y", visible=False)
    vs.despine(axL)
    vs.panel_title(axL, "Middle overs track winning most closely")

    axR.barh(y, d.edge_range, height=.30,
             color=[vs.S1_BLUE if p == "Middle" else vs.MUTED for p in d.phase],
             zorder=2)
    for i, (_, r) in zip(y, d.iterrows()):
        axR.text(r.edge_range + .15, i, f"{r.edge_range:.1f}", va="center",
                 fontsize=9.5, color=vs.INK_2)
    axR.set_yticks(y, [""] * len(d))
    axR.set_xlabel("Gap between best and worst team (runs/over)")
    axR.set_xlim(0, 6.6)
    axR.grid(axis="y", visible=False)
    vs.despine(axR, keep=("bottom",))
    vs.panel_title(axR, "but teams differ most at the death")

    vs.title_block(fig, "Where seasons are won is not where teams differ most",
                   "20 team-seasons, IPL 2023–24. Net edge = own scoring rate minus "
                   "rate conceded, against the league average for that phase.")
    vs.caption(fig, "The death overs offer the widest separation between teams and the weakest link "
                    "to winning a season — the noisiest phase to build a side around.")
    save(fig, "fig1_phase_vs_success.png")


# ---------------------------------------------------------------- figure 2
def fig2():
    edge = pd.read_parquet(PROC / "phase_edge.parquet")
    rec = pd.read_parquet(PROC / "season_records.parquet")
    m = edge[edge.phase == "Middle"].merge(rec, on=["year", "team"])

    champs = {(2023, "Chennai Super Kings"), (2024, "Kolkata Knight Riders")}
    runners = {(2023, "Gujarat Titans"), (2024, "Sunrisers Hyderabad")}

    def role(r):
        if (r.year, r.team) in champs: return "champion"
        if (r.year, r.team) in runners: return "runner-up"
        if r.team == RR: return "rr"
        return "other"

    m["role"] = m.apply(role, axis=1)
    style = {"champion": (vs.S1_BLUE, 160, "Champion"),
             "runner-up": (vs.S2_ORANGE, 125, "Runner-up"),
             "rr": (vs.S3_AQUA, 140, "Rajasthan Royals"),
             "other": (vs.MUTED, 58, "Other team-seasons")}
    # label offsets tuned to avoid collisions
    off = {("Chennai Super Kings", 2023): (12, -4), ("Kolkata Knight Riders", 2024): (-14, -20),
           ("Gujarat Titans", 2023): (12, -4), ("Sunrisers Hyderabad", 2024): (12, -4),
           (RR, 2023): (12, -4), (RR, 2024): (12, -4)}

    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    vs.layout(fig, top=0.80, bottom=0.235, left=0.095, right=0.97)

    z = np.polyfit(m.net_edge, m.win_pct, 1)
    xs = np.linspace(m.net_edge.min() - .25, m.net_edge.max() + .25, 50)
    ax.plot(xs, np.polyval(z, xs), color=vs.BASELINE, lw=1.4, zorder=1)
    ax.axvline(0, color=vs.GRID, lw=1, zorder=1)

    for r_ in ["other", "runner-up", "rr", "champion"]:
        s = m[m.role == r_]
        c, sz, _ = style[r_]
        ax.scatter(s.net_edge, s.win_pct, s=sz, color=c, zorder=3,
                   edgecolor=vs.SURFACE, linewidth=2, alpha=.8 if r_ == "other" else 1)

    for _, r in m[m.role != "other"].iterrows():
        name = "RR" if r.team == RR else r.team.split()[0]
        ax.annotate(f"{name} {r.year}", (r.net_edge, r.win_pct),
                    textcoords="offset points", xytext=off[(r.team, r.year)],
                    fontsize=9.5, color=vs.INK_2)

    ax.set_xlabel("Net middle-overs edge   (runs per over above league average)")
    ax.set_ylabel("Season win %")
    ax.set_xlim(-1.25, 1.95)
    vs.despine(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markersize=9,
                              markerfacecolor=style[k][0], markeredgecolor=vs.SURFACE,
                              markeredgewidth=1.5, label=style[k][2])
                       for k in ["champion", "runner-up", "rr", "other"]],
              loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.145),
              columnspacing=1.4, handletextpad=0.4)

    vs.title_block(fig, "Both champions led the league in the middle overs",
                   "Each point is one team-season (n = 20).")
    vs.caption(fig, "Both runners-up reached the final on a bookend phase instead — GT ranked 1st at the death, "
                    "SRH 1st in the powerplay —\nand both lost to the side that owned overs 7–15.", y=0.045)
    save(fig, "fig2_champion_validation.png")


# ---------------------------------------------------------------- figure 3
def fig3():
    tp = pd.read_parquet(PROC / "team_phase.parquet")
    # directional metrics only: control is style, not quality (Block 2)
    rks = ["bat_sr_rk", "bat_bnd_pct_rk", "bat_dot_pct_rk", "bat_bpd_rk",
           "bowl_econ_rk", "bowl_dot_pct_rk", "bowl_bpw_rk"]
    labels = ["Strike\nrate", "Boundary\n%", "Dot %\n(low good)", "Balls per\ndismissal",
              "Economy", "Dot %\nconceded", "Balls per\nwicket"]
    s = tp[tp.team == RR].set_index(["year", "phase"])[rks]
    s = s.reindex(pd.MultiIndex.from_product([[2023, 2024], PHASES], names=["year", "phase"]))
    mat = s.values.astype(float)

    cmap = LinearSegmentedColormap.from_list("rank_div", [vs.DIV_GOOD, vs.DIV_MID, vs.DIV_BAD])
    norm = TwoSlopeNorm(vmin=1, vcenter=5.5, vmax=10)

    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    vs.layout(fig, top=0.76, bottom=0.17, left=0.145, right=0.745)
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=11.5,
                    color="#ffffff" if (v <= 3 or v >= 8) else vs.INK, fontweight="bold")
    ax.set_xticks(range(len(labels)), labels, fontsize=9.5, linespacing=1.35)
    ax.set_yticks(range(len(s)), [f"{y}   {p}" for y, p in s.index], fontsize=10.5)
    ax.tick_params(axis="y", labelcolor=vs.INK)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(s), 1), minor=True)
    ax.grid(which="minor", color=vs.SURFACE, linewidth=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.add_patch(plt.Rectangle((-.5, .5), len(labels), 1, fill=False,
                               edgecolor=vs.INK, lw=2.4, zorder=5))
    ax.text(len(labels) - .34, 1.5, "bottom-half\non 7 of 7", va="center",
            ha="left", fontsize=10.5, color=vs.INK, fontweight="bold", clip_on=False)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=.030, pad=.26, ticks=[1, 5.5, 10])
    cb.ax.set_yticklabels(["1  best", "median", "10  worst"], fontsize=9)
    cb.outline.set_visible(False)

    vs.title_block(fig, "Rajasthan Royals' league rank on every directional metric",
                   "1 = best of 10. Every other phase-season is bottom-half on at most 3 of 7.\n"
                   "Control is excluded: it measures risk appetite, not quality.")
    save(fig, "fig3_rr_rank_heatmap.png")


# ---------------------------------------------------------------- figure 4
def fig4():
    d = pd.read_csv(PROC / "rr_significance.csv")
    d = d[d.season != "POOLED"].copy()
    d["phase"] = pd.Categorical(d.phase, PHASES, ordered=True)
    d = d.sort_values(["phase", "season"], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    vs.layout(fig, top=0.76, bottom=0.265, left=0.17, right=0.965)

    ax.axvline(0, color=vs.INK, lw=1.3, zorder=2)
    for i, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=vs.S1_BLUE, lw=2.6,
                solid_capstyle="round", alpha=.40, zorder=3)
        ax.scatter([r["diff"]], [i], s=100, color=vs.S1_BLUE, zorder=4,
                   edgecolor=vs.SURFACE, linewidth=2)
    ax.set_yticks(y, [f"{r.phase}   {r.season}" for _, r in d.iterrows()])
    ax.tick_params(axis="y", labelsize=10.5, labelcolor=vs.INK)
    ax.set_xlabel("RR batting strike rate minus the other nine teams\n"
                  "cluster bootstrap by match · 10,000 resamples · 95% CI")
    ax.set_xlim(-46, 54)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.grid(axis="y", visible=False)
    vs.despine(ax)
    ax.text(2, -0.55, "RR better →", fontsize=9, color=vs.MUTED, va="center")
    ax.text(-2, -0.55, "← RR worse", fontsize=9, color=vs.MUTED, ha="right", va="center")

    vs.title_block(fig, "Every interval crosses zero: no single-season gap is significant",
                   "With 14–15 matches a season, team phase splits cannot be resolved.")
    vs.caption(fig, "This is why the argument rests on a pattern across seven metrics and 20 team-seasons, "
                    "not on any one strike-rate split.")
    save(fig, "fig4_uncertainty.png")


if __name__ == "__main__":
    print("Writing figures to reports/figures/")
    fig1(); fig2(); fig3(); fig4()
    print("Block 3 complete.")
