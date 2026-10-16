"""Block 1: clean the raw dataset into data/processed/clean.parquet and matches.parquet."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data_Premier_League_Prelims_Dataset.csv"
OUT = ROOT / "data" / "processed"

# Phase boundaries (standard T20): powerplay 1-6, middle 7-15, death 16-20.
PHASE_BINS = {"Powerplay": (1, 6), "Middle": (7, 15), "Death": (16, 20)}

# Dismissals not credited to the bowler.
NON_BOWLER_DISMISSALS = {
    "run out",
    "retired out",
    "retired not out (hurt)",
    "obstructing the field",
}

# winner/toss spell RCB "Bangalore" in 2023 while team_bat/team_bowl say "Bengaluru"; unify them.
TEAM_ALIASES = {"Royal Challengers Bangalore": "Royal Challengers Bengaluru"}

TEAM_COLS = ["team_bat", "team_bowl", "winner", "toss"]


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_raw():
    if not RAW.exists():
        sys.exit(f"ERROR: raw dataset not found at {RAW}")
    df = pd.read_csv(RAW, low_memory=False)
    print(f"Loaded {RAW.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# --- Fix 1: team-name normalisation ---------------------------------------
def fix_team_names(df):
    rule("FIX 1  Team-name normalisation")
    before = {c: sorted(df[c].dropna().unique()) for c in TEAM_COLS}
    n_rcb_old = int((df[TEAM_COLS] == "Royal Challengers Bangalore").sum().sum())

    for c in TEAM_COLS:
        df[c] = df[c].replace(TEAM_ALIASES)

    print(f"  Replaced 'Royal Challengers Bangalore' -> 'Bengaluru' in {n_rcb_old:,} cells")
    for c in TEAM_COLS:
        print(f"  {c:10s}: {len(before[c])} distinct -> {df[c].nunique()} distinct")

    # Prove the bug this fixes: RCB's 2023 wins are invisible without it.
    rcb = "Royal Challengers Bengaluru"
    raw = pd.read_csv(RAW, low_memory=False, usecols=["p_match", "year", "winner"])
    broken = raw[(raw.year == 2023) & (raw.winner == rcb)].p_match.nunique()
    fixed = df[(df.year == 2023) & (df.winner == rcb)].p_match.nunique()
    print(f"  IMPACT: RCB matches won in 2023 -- before fix: {broken}, after fix: {fixed}")
    return df


# --- Fix 2: no-result match ------------------------------------------------
def flag_no_result(df):
    rule("FIX 2  No-result match")
    nr = sorted(df.loc[df.winner == "-", "p_match"].unique())
    df["no_result"] = df.p_match.isin(nr)
    df["winner"] = df.winner.replace("-", pd.NA)
    for m in nr:
        s = df[df.p_match == m]
        print(f"  p_match {m} | {s.match_date.iloc[0]} | {s.ground.iloc[0]}")
        print(f"    innings present: {sorted(s.inns.unique())}, balls: {len(s)} -> winner set to NA")
    print(f"  Flagged {len(nr)} match ({int(df.no_result.sum()):,} deliveries) as no_result")
    return df


# --- Fix 3: bowler-credited wickets ---------------------------------------
def add_bowler_wicket(df):
    rule("FIX 3  Dismissals and bowler-credited wickets")
    # Retired hurt is not a dismissal: reset 'out' where the wicket counter does not move.
    d = df.sort_values(["p_match", "inns", "ball_id", "row_id"])
    prev = d.groupby(["p_match", "inns"]).inns_wkts.shift(1).fillna(0)
    rh = d.index[(d.dismissal == "retired not out (hurt)") & d.out & (d.inns_wkts == prev)]
    df.loc[rh, "out"] = False
    print(f"  Retired-hurt events reset to not out (wicket counter unchanged): {len(rh)}")
    df["bowler_wicket"] = df.out & ~df.dismissal.isin(NON_BOWLER_DISMISSALS)
    # Striker dismissals only: bat_out is false when the non-striker is out.
    df["striker_dismissed"] = df.out & df.bat_out

    total = int(df.out.sum())
    bowler = int(df.bowler_wicket.sum())
    print(f"  Total dismissals          : {total:,}")
    print(f"  Credited to bowler        : {bowler:,}")
    print(f"  NOT credited to bowler    : {total - bowler:,}")
    print(df[df.out & ~df.bowler_wicket].dismissal.value_counts().to_string().replace("\n", "\n    "))
    ns = int((df.out & ~df.bat_out).sum())
    print(f"  Dismissals where striker was NOT the player out: {ns}")
    return df


# --- Fix 4: phase ----------------------------------------------------------
def add_phase(df):
    rule("FIX 4  Phase column")
    conds = [df.over_num.between(*PHASE_BINS["Powerplay"]), df.over_num.between(*PHASE_BINS["Middle"])]
    df["phase"] = np.select(conds, ["Powerplay", "Middle"], default="Death")
    df["phase"] = pd.Categorical(df.phase, ["Powerplay", "Middle", "Death"], ordered=True)
    for p, (lo, hi) in PHASE_BINS.items():
        n = int((df.phase == p).sum())
        print(f"  {p:10s} overs {lo:2d}-{hi:2d}: {n:,} deliveries ({100 * n / len(df):.1f}%)")
    return df


# --- Fix 5: legal deliveries ----------------------------------------------
def add_legal_ball(df):
    rule("FIX 5  Legal-delivery flag")
    # ballfaced is 0 only on wides; no-balls are faced but are not legal deliveries.
    df["legal_ball"] = (df.wide == 0) & (df.noball == 0)
    print(f"  Wides              : {int((df.wide > 0).sum()):,}")
    print(f"  No-balls           : {int((df.noball > 0).sum()):,}")
    print(f"  Legal deliveries   : {int(df.legal_ball.sum()):,} of {len(df):,}")
    mismatch = int(((df.ballfaced == 0) != (df.wide > 0)).sum())
    print(f"  Cross-check: rows where ballfaced==0 but not a wide: {mismatch}")
    return df


# --- Fix 6: rain-affected matches -----------------------------------------
def flag_shortened(df):
    rule("FIX 6  Rain-affected / shortened matches")
    # Flag the whole match if any delivery has a non-standard ball limit.
    affected = df.max_balls.ne(120)
    df["shortened"] = affected.groupby(df.p_match).transform("any")
    print("  max_balls value counts:")
    print(df.max_balls.value_counts().to_string().replace("\n", "\n    "))
    aff = df[affected].groupby(["p_match", "match_date", "ground", "max_balls"]).size()
    print("\n  Deliveries identifying affected matches:")
    print(aff.to_string().replace("\n", "\n    "))
    print("\n  NOTE: max_balls == 0 is a raw-data artefact, not a real limit.")
    print(f"  Flagged {int(df.shortened.sum()):,} deliveries across "
          f"{df[df.shortened].p_match.nunique()} matches.")
    print(f"  Includes {int((df.shortened & ~affected).sum())} deliveries with max_balls=120 "
          "in those same matches.")
    return df


# --- Fix 7: match-level table ---------------------------------------------
def build_matches(df):
    rule("FIX 7  Match-level table")
    m = (df.groupby("p_match")
           .agg(match_date=("match_date", "first"),
                year=("year", "first"),
                ground=("ground", "first"),
                winner=("winner", "first"),
                toss=("toss", "first"),
                no_result=("no_result", "first"),
                shortened=("shortened", "any"),
                balls=("row_id", "size"),
                innings=("inns", "nunique"))
           .reset_index())

    sides = (pd.concat([df[["p_match", "team_bat"]].rename(columns={"team_bat": "team"}),
                        df[["p_match", "team_bowl"]].rename(columns={"team_bowl": "team"})])
             .drop_duplicates())
    pair = sides.groupby("p_match").team.apply(lambda s: sorted(s))
    m["team_a"] = m.p_match.map(lambda i: pair[i][0])
    m["team_b"] = m.p_match.map(lambda i: pair[i][1] if len(pair[i]) > 1 else pd.NA)

    # Who batted first.
    first = df[df.inns == 1].groupby("p_match").team_bat.first()
    m["bat_first"] = m.p_match.map(first)
    m["toss_won_by"] = m.toss
    m["result"] = np.where(m.no_result, "no result",
                           np.where(m.winner == m.bat_first, "defended", "chased"))

    m = m.sort_values("match_date").reset_index(drop=True)
    print(f"  Built {len(m)} matches x {m.shape[1]} columns")
    print(f"  Matches per season: {m.groupby('year').size().to_dict()}")
    print(f"  Results: {m.result.value_counts().to_dict()}")
    return m


# --- Fix 8: run reconciliation --------------------------------------------
def fix_delivery_runs(df):
    """Correct run fields from the cumulative counters; stop if the counters disagree."""
    rule("FIX 8  Delivery run corrections")
    ordered = df.sort_values(["p_match", "inns", "ball_id"])
    score = ordered.batruns + ordered.wide + ordered.noball + ordered.byes + ordered.legbyes
    bowlruns = ordered.batruns + ordered.wide + ordered.noball
    innings_delta = (ordered.groupby(["p_match", "inns"]).inns_runs.diff()
                     .fillna(ordered.inns_runs))
    bowler_delta = (ordered.groupby(["p_match", "inns", "p_bowl"]).cur_bowl_runs.diff()
                    .fillna(ordered.cur_bowl_runs))
    if not (score.eq(innings_delta).all() and bowlruns.eq(bowler_delta).all()):
        raise ValueError("Run components disagree with cumulative innings/bowler counters; "
                         "review the raw deliveries before correcting them.")

    changed = ordered.score.ne(score) | ordered.bowlruns.ne(bowlruns)
    audit = ordered.loc[changed, ["row_id", "p_match", "inns", "ball_id",
                                 "batruns", "wide", "noball", "byes", "legbyes",
                                 "score", "bowlruns"]].copy()
    audit = audit.rename(columns={"score": "raw_score", "bowlruns": "raw_bowlruns"})
    audit["corrected_score"] = score.loc[changed]
    audit["corrected_bowlruns"] = bowlruns.loc[changed]
    print(f"  Corrected deliveries: {len(audit)}")
    print(f"  Runs restored to innings totals: "
          f"{int((audit.corrected_score - audit.raw_score).sum())}")
    print(f"  Runs restored to bowler totals: "
          f"{int((audit.corrected_bowlruns - audit.raw_bowlruns).sum())}")
    print("  Checked against cumulative innings and per-bowler runs on every delivery.")
    df.loc[ordered.index, "score"] = score
    df.loc[ordered.index, "bowlruns"] = bowlruns
    return audit.reset_index(drop=True)


def reconcile_runs(df):
    rule("INNINGS RUN RECONCILIATION")
    chk = (df.sort_values(["p_match", "inns", "ball_id"])
             .groupby(["p_match", "inns"])
             .agg(summed=("score", "sum"), reported=("inns_runs", "last"))
             .reset_index())
    chk["diff"] = chk.reported - chk.summed
    bad = chk[chk["diff"] != 0]
    print(f"  Innings checked: {len(chk)}")
    print(f"  Innings where sum(score) != final inns_runs: {len(bad)}")
    if len(bad):
        print(f"  Discrepancy range: {bad['diff'].min():+d} to {bad['diff'].max():+d} runs")
        print(bad.to_string(index=False).replace("\n", "\n    "))
        raise ValueError("Corrected delivery scores do not reconcile with innings totals.")
    print("  All innings reconcile exactly after the documented delivery corrections.")
    return chk


# --- Extra validation ------------------------------------------------------
def validate(df, matches):
    rule("VALIDATION")
    print(f"  Deliveries            : {len(df):,}")
    print(f"  Matches               : {df.p_match.nunique()}")
    print(f"  Seasons               : {sorted(df.year.unique())}")
    print(f"  Teams                 : {df.team_bat.nunique()}")
    print(f"  Batters / bowlers     : {df.p_bat.nunique()} / {df.p_bowl.nunique()}")
    print(f"  Grounds               : {df.ground.nunique()}")
    print(f"  Duplicate row_id      : {int(df.row_id.duplicated().sum())}")
    print(f"  Nulls in key columns  : "
          f"{int(df[['p_match', 'inns', 'over_num', 'team_bat', 'team_bowl', 'score']].isna().sum().sum())}")

    # Arithmetic identities stated in the data dictionary.
    id1 = (df.score == df.batruns + df.wide + df.noball + df.byes + df.legbyes).mean()
    id2 = (df.bowlruns == df.batruns + df.wide + df.noball).mean()
    print(f"  score == batruns+extras            : {100 * id1:.2f}% of rows")
    print(f"  bowlruns == batruns+wide+noball    : {100 * id2:.2f}% of rows")

    # Is bowl_release_point a per-ball or a per-bowler attribute?
    per_bowler = df.groupby("p_bowl").bowl_release_point.nunique(dropna=True)
    print(f"  bowl_release_point: bowlers with >1 distinct value: "
          f"{int((per_bowler > 1).sum())} of {len(per_bowler)} -- treat as per-delivery")

    # Expected missing matches (abandoned without play).
    per_team = (pd.concat([matches[['year', 'team_a']].rename(columns={'team_a': 't'}),
                           matches[['year', 'team_b']].rename(columns={'team_b': 't'})])
                .dropna().groupby(['year', 't']).size().unstack(0))
    print("\n  Matches per team per season (2024 gaps = 3 abandoned games, no play):")
    print(per_team.to_string().replace("\n", "\n    "))


def main():
    df = load_raw()
    df = fix_team_names(df)
    df = flag_no_result(df)
    df = add_bowler_wicket(df)
    df = add_phase(df)
    df = add_legal_ball(df)
    df = flag_shortened(df)
    matches = build_matches(df)
    corrections = fix_delivery_runs(df)
    recon = reconcile_runs(df)
    validate(df, matches)

    rule("WRITING OUTPUT")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "clean.parquet", index=False)
    matches.to_parquet(OUT / "matches.parquet", index=False)
    recon.to_csv(OUT / "run_reconciliation.csv", index=False)
    corrections.to_csv(OUT / "run_corrections.csv", index=False)
    for f in ["clean.parquet", "matches.parquet", "run_reconciliation.csv", "run_corrections.csv"]:
        print(f"  {f:28s} {(OUT / f).stat().st_size / 1024:>8.0f} KB")
    print("\nBlock 1 complete.")


if __name__ == "__main__":
    main()
