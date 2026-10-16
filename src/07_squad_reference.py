"""Block 7: RR squad reference with sourced nationality and keeper roles."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "reference"
RR = "Rajasthan Royals"

AUCTION_SOURCE = "IPL_Auction_2023_24.xlsx"
MAHARAJ_SOURCE = "https://www.rajasthanroyals.com/latest-news/keshav-maharaj-replacement-prasidh-krishna-rr-ipl-2024"


def squad_metadata():
    """Read nationality and keeper roles from the auction file, with one cited replacement."""
    sheets = pd.read_excel(ROOT / AUCTION_SOURCE, sheet_name=["IPL 2023", "IPL 2024"], header=1)
    a = pd.concat(sheets.values(), ignore_index=True)
    a["Player Name"] = a["Player Name"].replace({"Donavon Ferreira": "Donovan Ferreira"})
    meta = a.groupby("Player Name").agg(
        nationality=("Country", "last"), keeper=("Type", lambda t: "WK-Batter" in set(t)))
    meta["metadata_source"] = AUCTION_SOURCE
    meta.loc["Keshav Maharaj"] = ["South Africa", False, MAHARAJ_SOURCE]
    return meta.rename_axis("player").reset_index()


def derive_role(r):
    """Assign a role from balls batted and bowled in the dataset."""
    bats = r.balls >= 100
    bowls = r.bowl_balls >= 100
    if bats and bowls:
        return "All-rounder"
    if bowls:
        return "Bowler"
    if bats:
        return "Wicketkeeper-batter" if r.keeper else "Batter"
    return "Fringe / limited appearances"


def main():
    d = pd.read_parquet(PROC / "clean.parquet")

    bat = (d[d.team_bat == RR].groupby(["p_bat", "bat"])
             .agg(balls=("ballfaced", "sum"), runs=("batruns", "sum"),
                  bat_hand=("bat_hand", "first"))
             .reset_index().rename(columns={"p_bat": "pid", "bat": "player"}))
    bowl = (d[d.team_bowl == RR].groupby(["p_bowl", "bowl"])
              .agg(bowl_balls=("legal_ball", "sum"), bowl_runs=("bowlruns", "sum"),
                   wickets=("bowler_wicket", "sum"), bowl_style=("bowl_style", "first"),
                   bowl_kind=("bowl_kind", "first"))
              .reset_index().rename(columns={"p_bowl": "pid", "bowl": "player"}))
    # Seasons a player appeared in, batting or bowling.
    apps = pd.concat([d.loc[d.team_bat == RR, ["p_bat", "year"]].set_axis(["pid", "year"], axis=1),
                      d.loc[d.team_bowl == RR, ["p_bowl", "year"]].set_axis(["pid", "year"], axis=1)])
    seasons = apps.groupby("pid").year.agg(lambda s: "/".join(map(str, sorted(s.unique()))))

    sq = bat.merge(bowl, on=["pid", "player"], how="outer")
    for c in ["balls", "runs", "bowl_balls", "bowl_runs", "wickets"]:
        sq[c] = sq[c].fillna(0).astype(int)

    sq = sq.merge(squad_metadata(), on="player", how="left", validate="one_to_one")
    if sq[["nationality", "keeper", "metadata_source"]].isna().any().any():
        raise ValueError("Missing squad metadata; check the cited sources.")

    sq["overseas"] = np.where(sq.nationality == "India", "No", "Yes")
    sq["role"] = sq.apply(derive_role, axis=1)
    sq["seasons"] = sq.pid.map(seasons)
    sq["bat_sr"] = np.where(sq.balls > 0, (100 * sq.runs / sq.balls).round(1), np.nan)
    sq["bowl_econ"] = np.where(sq.bowl_balls > 0, (6 * sq.bowl_runs / sq.bowl_balls).round(2), np.nan)
    cols = ["pid", "player", "nationality", "overseas", "metadata_source", "role", "keeper", "seasons",
            "bat_hand", "balls", "runs", "bat_sr",
            "bowl_style", "bowl_kind", "bowl_balls", "wickets", "bowl_econ"]
    sq = sq[cols].sort_values(["balls", "bowl_balls"], ascending=False).reset_index(drop=True)

    print(f"{'=' * 78}\nSQUAD REFERENCE -- {len(sq)} players\n{'=' * 78}")
    print("  Nationality and keeper roles: auction file; RR's announcement for Keshav Maharaj.\n")
    show = sq[["player", "nationality", "overseas", "role", "seasons"]]
    print(show.to_string(index=False))

    print("\n  OVERSEAS PLAYERS (4 may play in any XI):")
    ov = sq[(sq.overseas == "Yes")].sort_values("balls", ascending=False)
    for _, r in ov.iterrows():
        used = f"{int(r.balls)} balls batted" if r.balls > 0 else f"{int(r.bowl_balls)} balls bowled"
        print(f"    {r.player:22s} {r.nationality:14s} "
              f"{r.role:28s} {used}")
    print(f"\n  {len(ov)} overseas players used across the two seasons, "
          f"for 4 slots per XI.")

    REF.mkdir(parents=True, exist_ok=True)
    out = REF / "rr_squad_reference.csv"
    sq.to_csv(out, index=False)
    print(f"\n  Written: {out.relative_to(ROOT)}")
    print("  Each row records its nationality and keeper-role source.")


if __name__ == "__main__":
    main()
