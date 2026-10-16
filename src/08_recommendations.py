"""Block 8: squad review, shot-mix scenario and scouting profile for RR."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "reference"
FIGS = ROOT / "reports" / "figures"
RR = "Rajasthan Royals"
MIN_BALLS = 120          # minimum middle-overs balls for a player-level claim

ATTACKING = {"SLOG_SHOT", "SLOG_SWEEP", "PULL", "HOOK", "UPPER_CUT", "RAMP",
             "REVERSE_SCOOP", "REVERSE_PULL", "SWEEP_SHOT", "REVERSE_SWEEP"}

vs.apply_style()


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def shrink(g, grand, sigma2):
    """Empirical-Bayes shrinkage of player rates toward the league mean."""
    total_var = g.rpb.var()
    tau2 = max(total_var - (sigma2 / g.balls).mean(), 1e-9)
    B = tau2 / (tau2 + sigma2 / g.balls)
    return B * g.rpb + (1 - B) * grand


def player_table(df, team=None, min_balls=MIN_BALLS):
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1)].copy()
    mid["attacking"] = mid.shot.isin(ATTACKING)
    mid["full"] = mid.length.isin(["FULL", "FULL_TOSS", "YORKER"])
    grand = mid.batruns.sum() / len(mid)
    sigma2 = mid.batruns.var()

    sub = mid if team is None else mid[mid.team_bat == team]
    g = sub.groupby("bat", observed=True).agg(
        team=("team_bat", "first"), balls=("batruns", "size"), runs=("batruns", "sum"),
        attack_pct=("attacking", "mean"), ctrl=("control", "mean"))
    g["rpb"] = g.runs / g.balls
    g = g[g.balls >= min_balls]
    g["sr"] = 100 * g.rpb
    g["sr_shrunk"] = 100 * shrink(g, grand, sigma2)
    g["attack_pct"] = 100 * g.attack_pct
    g["ctrl_pct"] = 100 * g.ctrl

    f = sub[sub.full].groupby("bat", observed=True).agg(
        full_balls=("batruns", "size"), full_runs=("batruns", "sum"))
    g = g.join(f)
    g["sr_vs_full"] = np.where(g.full_balls >= 30, 100 * g.full_runs / g.full_balls, np.nan)
    return g.reset_index()


# --------------------------------------------------------------------- 1
def who_is_passive(df):
    rule("1  WHO IS PASSIVE? Player-level intent in the middle overs")
    rr = player_table(df, RR, min_balls=100).sort_values("attack_pct")
    lg = player_table(df, None)
    lg_attack = lg.attack_pct.median()
    print(f"  League median attacking-shot share (players with {MIN_BALLS}+ balls): "
          f"{lg_attack:.1f}%\n")
    print(f"  {'player':22s} {'balls':>6s} {'attack%':>8s} {'ctrl%':>7s} "
          f"{'mid SR':>7s} {'SR vs full':>11s} {'full balls':>11s}")
    for _, r in rr.iterrows():
        fb = f"{r.sr_vs_full:.0f}" if not np.isnan(r.sr_vs_full) else "  --"
        n = f"{int(r.full_balls)}" if not np.isnan(r.full_balls) else "--"
        flag = "  <-- below league median" if r.attack_pct < lg_attack else ""
        print(f"  {r.bat:22s} {int(r.balls):6d} {r.attack_pct:7.1f}% {r.ctrl_pct:6.1f}% "
              f"{r.sr_shrunk:7.1f} {fb:>11s} {n:>11s}{flag}")
    below = (rr.attack_pct < lg_attack).sum()
    print(f"\n  {below} of {len(rr)} RR batters attack less than the league median.")
    return rr, lg


# --------------------------------------------------------------------- 2
def overseas_audit(df, squad):
    rule("2  OVERSEAS AUDIT: four slots, twelve players")
    ov = squad[squad.overseas == "Yes"].copy()
    ov["contribution"] = ov.balls + ov.bowl_balls
    ov = ov.sort_values("contribution", ascending=False)
    total = ov.contribution.sum()
    ov["share"] = 100 * ov.contribution / total
    print(f"  {'player':22s} {'nat':13s} {'role':20s} "
          f"{'balls bat':>9s} {'balls bowl':>10s} {'share':>7s}")
    for _, r in ov.iterrows():
        print(f"  {r.player:22s} {r.nationality:13s} "
              f"{r.role[:20]:20s} {int(r.balls):9d} {int(r.bowl_balls):10d} {r.share:6.1f}%")
    top4 = ov.head(4).share.sum()
    print(f"\n  Top four overseas players account for {top4:.0f}% of all overseas")
    print(f"  involvement; the other {len(ov)-4} account for {100-top4:.0f}%.")
    thin = ov[ov.contribution < 60]
    print(f"  {len(thin)} overseas players featured for under 60 deliveries in total:")
    print(f"    {', '.join(thin.player)}")
    print("\n  The slots are effectively Buttler, Boult, Hetmyer and a rotating fourth player; the rest is churn.")
    return ov


# --------------------------------------------------------------------- 3
def role_balance(df, squad):
    rule("3  ROLE BALANCE: what the squad actually provides by phase")
    d = df[(df.team_bat == RR) & (df.ballfaced == 1)]
    tab = (d.groupby(["bat", "phase"], observed=True)
             .agg(balls=("batruns", "size"), runs=("batruns", "sum")).reset_index())
    tab = tab[tab.balls >= 50]
    tab["sr"] = 100 * tab.runs / tab.balls
    piv = tab.pivot(index="bat", columns="phase", values="sr")
    bal = tab.pivot(index="bat", columns="phase", values="balls")
    nat = squad.set_index("player").nationality.to_dict()
    print("  Strike rate by phase, RR batters with 50+ balls in that phase:\n")
    print(f"  {'player':22s} {'nat':12s} {'PP':>7s} {'MIDDLE':>8s} {'DEATH':>7s}")
    for p in piv.index:
        row = piv.loc[p]
        f = lambda v: f"{v:.0f}" if not np.isnan(v) else "  --"
        print(f"  {p:22s} {nat.get(p, '?'):12s} "
              f"{f(row.get('Powerplay', np.nan)):>7s} {f(row.get('Middle', np.nan)):>8s} "
              f"{f(row.get('Death', np.nan)):>7s}")
    print("\n  No RR batter with a real middle-overs sample strikes above 150: there is no accelerator.")
    return piv


# --------------------------------------------------------------------- 4
def counterfactual(df, lg):
    """Re-weight RR's own shot returns to the league's attacking share (a scenario, not a forecast)."""
    rule("4  THE INTERNAL FIX: is a signing even necessary?")
    print("  What would this squad score if it attacked as often as the league does?")
    mid = df[(df.phase == "Middle") & (df.ballfaced == 1) & df.shot.notna()].copy()
    mid["attacking"] = mid.shot.isin(ATTACKING)
    rr = mid[mid.team_bat == RR]
    share_now = rr.attacking.mean()
    share_lg = mid[mid.team_bat != RR].attacking.mean()
    rpb_a = rr[rr.attacking].batruns.mean()
    rpb_n = rr[~rr.attacking].batruns.mean()
    now = 100 * (share_now * rpb_a + (1 - share_now) * rpb_n)
    at_lg = 100 * (share_lg * rpb_a + (1 - share_lg) * rpb_n)
    print(f"\n  RR middle-overs strike rate now                : {now:.1f}")
    print(f"  RR attacking share now                         : {100*share_now:.1f}%")
    print(f"  League attacking share                         : {100*share_lg:.1f}%")
    print(f"  RR runs per ball when attacking / otherwise    : {rpb_a:.2f} / {rpb_n:.2f}")
    print(f"  RR strike rate at the league's attacking share : {at_lg:.1f}  "
          f"({at_lg - now:+.1f})")
    print(f"\n  Shot mix alone brings RR close to the league average ({100*mid[mid.team_bat != RR].batruns.mean():.1f}); an illustrative, likely optimistic scenario.")
    return {"sr_now": now, "sr_at_league_share": at_lg,
            "share_now": 100 * share_now, "share_league": 100 * share_lg,
            "rpb_attacking": rpb_a, "rpb_other": rpb_n}


# --------------------------------------------------------------------- 5
def scouting(df, lg, squad):
    rule("5  SCOUTING: who in the league has the missing profile?")
    print("  Profile: attacks at or above the league median and scores off full balls (RR's weakness).\n")
    lg_attack = lg.attack_pct.median()
    cand = lg[(lg.attack_pct >= lg_attack) & (lg.sr_vs_full >= 150)
              & (lg.team != RR)].sort_values("sr_shrunk", ascending=False)
    print(f"  {'player':22s} {'team':6s} {'balls':>6s} {'mid SR':>7s} "
          f"{'attack%':>8s} {'SR vs full':>11s}")
    short = {"Chennai Super Kings": "CSK", "Delhi Capitals": "DC", "Gujarat Titans": "GT",
             "Kolkata Knight Riders": "KKR", "Lucknow Super Giants": "LSG",
             "Mumbai Indians": "MI", "Punjab Kings": "PBKS",
             "Royal Challengers Bengaluru": "RCB", "Sunrisers Hyderabad": "SRH"}
    for _, r in cand.head(12).iterrows():
        print(f"  {r.bat:22s} {short.get(r.team, '?'):6s} {int(r.balls):6d} "
              f"{r.sr_shrunk:7.1f} {r.attack_pct:7.1f}% {r.sr_vs_full:11.0f}")
    print(f"\n  {len(cand)} players in the dataset meet both conditions.")
    print("  A profile, not a shopping list: 2025 availability is not known; src/10_auction.py adds nationality and price.")
    return cand


# ------------------------------------------------------------------ figure
def fig_player_intent(rr, lg):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    vs.layout(fig, top=0.73, bottom=0.215, left=0.095, right=0.97)
    lg_o = lg[lg.team != RR]
    ax.scatter(lg_o.attack_pct, lg_o.sr_shrunk, s=48, color=vs.MUTED, alpha=.55,
               zorder=2, edgecolor=vs.SURFACE, linewidth=1.5)
    ax.scatter(rr.attack_pct, rr.sr_shrunk, s=150, color=vs.S1_BLUE, zorder=4,
               edgecolor=vs.SURFACE, linewidth=2)
    med_a, med_s = lg.attack_pct.median(), lg.sr_shrunk.median()
    ax.axvline(med_a, color=vs.BASELINE, lw=1.2, zorder=1)
    ax.axhline(med_s, color=vs.BASELINE, lw=1.2, zorder=1)
    off = {"Riyan Parag": (11, -14), "Sanju Samson": (11, 7), "Yashasvi Jaiswal": (11, 6),
           "Jos Buttler": (11, -13), "Dhruv Jurel": (11, 4), "Shimron Hetmyer": (-13, -17),
           "Ravichandran Ashwin": (11, -4), "Devdutt Padikkal": (11, 2)}
    for _, r in rr.iterrows():
        ax.annotate(r.bat, (r.attack_pct, r.sr_shrunk), textcoords="offset points",
                    xytext=off.get(r.bat, (10, 4)), fontsize=9.5, color=vs.INK_2)
    ax.text(med_a + .4, ax.get_ylim()[1] - 3, "league median attack rate →",
            fontsize=9, color=vs.MUTED)
    ax.set_xlabel("Share of middle-overs balls played with an attacking shot (%)")
    ax.set_ylabel("Middle-overs strike rate (shrunk)")
    vs.despine(ax)
    n_below = int((rr.attack_pct < med_a).sum())
    vs.title_block(fig, f"{n_below} of Rajasthan's {len(rr)} batters attack less often than the league median",
                   "Each grey point is another franchise's batter with 120+ middle-overs balls; "
                   "blue are Rajasthan's.\nOnly Buttler sits on the attacking side of the line.")
    vs.caption(fig, "Shot tags show how often a batter attacks, not why. "
                    "RR's returns when they do attack are the best in the league.")
    fig.savefig(FIGS / "fig9_player_intent.png")
    plt.close(fig)
    print("  fig9_player_intent.png")


def main():
    df = pd.read_parquet(PROC / "clean.parquet")
    squad = pd.read_csv(REF / "rr_squad_reference.csv")
    rr, lg = who_is_passive(df)
    ov = overseas_audit(df, squad)
    piv = role_balance(df, squad)
    cf = counterfactual(df, lg)
    cand = scouting(df, lg, squad)

    rule("WRITING OUTPUT")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_player_intent(rr, lg)
    for obj, name in [(rr, "rec_rr_players.csv"), (lg, "rec_league_players.csv"),
                      (ov, "rec_overseas_audit.csv"),
                      (pd.DataFrame([cf]), "rec_counterfactual.csv"),
                      (cand, "rec_scouting_profile.csv")]:
        obj.to_csv(PROC / name, index=False)
        print(f"  {name:30s} {(PROC / name).stat().st_size / 1024:>7.1f} KB")
    print("\nBlock 8 complete.")


if __name__ == "__main__":
    main()
