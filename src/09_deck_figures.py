"""Block 9: render the two slide figures at the template panel sizes."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
RR = "Rajasthan Royals"
SHORT = {"Chennai Super Kings": "CSK", "Delhi Capitals": "DC", "Gujarat Titans": "GT",
         "Kolkata Knight Riders": "KKR", "Lucknow Super Giants": "LSG",
         "Mumbai Indians": "MI", "Punjab Kings": "PBKS", "Rajasthan Royals": "RR",
         "Royal Challengers Bengaluru": "RCB", "Sunrisers Hyderabad": "SRH"}
CHAMP = {2023: "Chennai Super Kings", 2024: "Kolkata Knight Riders"}

# Match the template's panel colour so each figure blends into its panel.
PANEL = "#F7F5F2"
vs.apply_style()
plt.rcParams.update({"savefig.dpi": 300, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "axes.labelsize": 10, "figure.facecolor": PANEL,
                     "axes.facecolor": PANEL, "savefig.facecolor": PANEL})


def s2_framing():
    """Slide 2: season win rates with RR and the champion highlighted."""
    rec = pd.read_parquet(PROC / "season_records.parquet")
    fig, axes = plt.subplots(2, 1, figsize=(5.84, 4.71))
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.13, right=0.97, hspace=0.62)
    for ax, y in zip(axes, [2023, 2024]):
        r = rec[rec.year == y].sort_values("win_pct", ascending=False).reset_index(drop=True)
        x = np.arange(len(r))
        cols = [vs.S1_BLUE if t == RR else vs.S2_ORANGE if t == CHAMP[y] else vs.MUTED
                for t in r.team]
        ax.bar(x, r.win_pct, width=.62, color=cols, zorder=2)
        for i, (t, w) in enumerate(zip(r.team, r.win_pct)):
            if t in (RR, CHAMP[y]):
                ax.text(i, w + 2.5, f"{w:.0f}%", ha="center", fontsize=10,
                        fontweight="bold", color=vs.INK)
        ax.set_xticks(x, [SHORT[t] for t in r.team], fontsize=9.5)
        ax.set_ylim(0, 95)
        ax.set_yticks([0, 25, 50, 75])
        ax.set_ylabel("Win %", fontsize=10)
        ax.grid(axis="x", visible=False)
        vs.despine(ax)
        # Rank with ties stated honestly: RR's 50% in 2023 equals RCB's.
        rr_w = float(r.loc[r.team == RR, "win_pct"].iloc[0])
        above = int((r.win_pct > rr_w + 1e-9).sum())
        tied = int((abs(r.win_pct - rr_w) < 1e-9).sum()) > 1
        n = above + 1
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n, "th")
        rk = f"{'joint ' if tied else ''}{n}{suf}"
        ax.text(0, 1.06, f"{y}", transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=vs.INK, ha="left", va="bottom")
        ax.text(0.12, 1.06, f"RR {rk} of 10 by win rate  ·  champion {SHORT[CHAMP[y]]}",
                transform=ax.transAxes, fontsize=10, color=vs.INK_2, ha="left", va="bottom")
    fig.text(0.02, 0.965, "Second-best record in 2024 — and no final", fontsize=13.5,
             fontweight="bold", color=vs.INK, ha="left", va="top")
    fig.savefig(FIGS / "deck_s2_framing.png")
    plt.close(fig)
    print("  deck_s2_framing.png      5.84 x 4.71 in")


def s4_overview():
    """Slide 4: RR vs each season's champion, league rank by phase."""
    tp = pd.read_parquet(PROC / "team_phase.parquet")
    rows = [("bat_sr_rk", "Powerplay", "Batting · powerplay"),
            ("bat_sr_rk", "Middle", "Batting · middle"),
            ("bat_sr_rk", "Death", "Batting · death"),
            ("bowl_econ_rk", "Powerplay", "Bowling · powerplay"),
            ("bowl_econ_rk", "Middle", "Bowling · middle"),
            ("bowl_econ_rk", "Death", "Bowling · death")]
    fig, axes = plt.subplots(1, 2, figsize=(6.94, 4.66), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.27, right=0.975, wspace=0.10)
    y = np.arange(len(rows))[::-1]
    for ax, yr in zip(axes, [2023, 2024]):
        champ = CHAMP[yr]
        for yi, (col, ph, lab) in zip(y, rows):
            g = tp[(tp.year == yr) & (tp.phase == ph)].set_index("team")[col]
            rr_v, ch_v = g[RR], g[champ]
            mid = ph == "Middle"
            if mid:
                ax.axhspan(yi - .45, yi + .45, color=vs.GRID, alpha=.55, zorder=0, lw=0)
            ax.plot([rr_v, ch_v], [yi, yi], color=vs.BASELINE, lw=1.6, zorder=1)
            ax.scatter([ch_v], [yi], s=90, color=vs.S2_ORANGE, zorder=3,
                       edgecolor=PANEL, linewidth=1.8)
            ax.scatter([rr_v], [yi], s=90, color=vs.S1_BLUE, zorder=4,
                       edgecolor=PANEL, linewidth=1.8)
        ax.set_xlim(10.6, 0.4)                     # rank 1 (best) on the right
        ax.set_xticks([10, 7, 4, 1])
        ax.set_xlabel("League rank  (1 = best)", fontsize=9.5)
        ax.grid(axis="y", visible=False)
        vs.despine(ax, keep=("bottom",))
        ax.text(0.5, 1.035, f"{yr}  ·  vs {SHORT[champ]}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=vs.INK, ha="center", va="bottom")
    axes[0].set_yticks(y, [r[2] for r in rows], fontsize=10)
    axes[0].tick_params(axis="y", labelcolor=vs.INK)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, mfc=vs.S1_BLUE,
                               mec=PANEL, label="Rajasthan Royals"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc=vs.S2_ORANGE,
                               mec=PANEL, label="Season champion")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.62, -0.005),
               fontsize=9.5, frameon=False)
    fig.text(0.02, 0.965, "Champions own the middle overs; RR do not", fontsize=13,
             fontweight="bold", color=vs.INK, ha="left", va="top")
    fig.text(0.02, 0.895, "Batting = strike rate rank, bowling = economy rank. Shaded rows: overs 7–15.",
             fontsize=9, color=vs.INK_2, ha="left", va="top")
    fig.savefig(FIGS / "deck_s4_overview.png")
    plt.close(fig)
    print("  deck_s4_overview.png     6.94 x 4.66 in")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    print("Rendering slide-sized figures")
    s2_framing()
    s4_overview()
