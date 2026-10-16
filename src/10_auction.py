"""Block 10: RR's contracts against deliveries, and nationality and price for the scouting profile."""

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "IPL_Auction_2023_24.xlsx"
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "reference"
RR = "Rajasthan Royals"
MIN_DELIVERIES = 60       # same threshold as the overseas audit in Block 8

TEAMS = {"CSK": "Chennai Super Kings", "DC": "Delhi Capitals", "GT": "Gujarat Titans",
         "KKR": "Kolkata Knight Riders", "LSG": "Lucknow Super Giants", "MI": "Mumbai Indians",
         "PBKS": "Punjab Kings", "RR": "Rajasthan Royals", "RCB": "Royal Challengers Bengaluru",
         "SRH": "Sunrisers Hyderabad"}

# The auction file spells two players differently from the delivery data.
NAME_FIXES = {"Donavon Ferreira": "Donovan Ferreira", "Philip Salt": "Phil Salt"}


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def crore(v):
    """Convert '20.00 L' or '2.00 Cr' to crore; anything else is missing."""
    m = re.match(r"\s*([\d.]+)\s*(Cr|L)\b", str(v))
    return float(m.group(1)) / (1 if m.group(2) == "Cr" else 100) if m else np.nan


def load():
    """Both seasons as one table, one row per player and season."""
    rule("AUCTION FILE")
    if not RAW.exists():
        sys.exit(f"ERROR: auction file not found at {RAW}")
    seasons = []
    for year in (2023, 2024):
        s = pd.read_excel(RAW, sheet_name=f"IPL {year}", header=1)
        s.columns = ["n", "player", "type", "country", "status", "base", "final", "team"]
        seasons.append(s.drop(columns="n").assign(year=year))
    a = pd.concat(seasons, ignore_index=True)
    for y, s in a.groupby("year"):
        print(f"  {y}: {len(s)} rows; " + ", ".join(f"{k} {v}" for k, v in s.status.value_counts().items()))

    a["player"] = a.player.replace(NAME_FIXES)
    a["price_cr"] = a.final.map(crore)
    a["team"] = a.team.map(TEAMS)                 # Unsold becomes missing
    # One player is listed as both sold and unsold in 2024; keep the sale.
    dup = a.duplicated(["year", "player"], keep=False) & a.status.eq("Unsold")
    print(f"  Unsold duplicates dropped: {', '.join(f'{p} ({y})' for p, y in zip(a[dup].player, a[dup].year))}")
    a = a[~dup].reset_index(drop=True)
    print(f"  Names aligned to the delivery data: {', '.join(f'{k} -> {v}' for k, v in NAME_FIXES.items())}")
    print(f"  Squad rows without a price: {int(a[a.team.notna()].price_cr.isna().sum())}")
    print(f"  Rows without a country: {int(a.country.isna().sum())}")
    return a


# --------------------------------------------------------------------- 1
def rr_purse(a, d):
    """Each RR contract against the deliveries that player batted or bowled that season."""
    rule("1  RR CONTRACTS: price against deliveries, by season")
    bat = d[d.team_bat == RR].groupby(["year", "bat"]).ballfaced.sum().rename_axis(["year", "player"])
    bowl = d[d.team_bowl == RR].groupby(["year", "bowl"]).legal_ball.sum().rename_axis(["year", "player"])
    done = pd.concat([bat, bowl]).groupby(level=[0, 1]).sum()      # balls batted plus balls bowled

    rr = a[a.team == RR].copy()
    rr["deliveries"] = [int(done.get((y, p), 0)) for y, p in zip(rr.year, rr.player)]
    rr["overseas"] = rr.country.ne("India")
    for y, s in rr.groupby("year"):
        idle = s[s.deliveries == 0].sort_values("price_cr", ascending=False)
        low = s[s.deliveries < MIN_DELIVERIES]
        on_field = set(done.loc[y][done.loc[y] > 0].index)
        print(f"\n  {y}: {s.price_cr.sum():.2f} Cr across {len(s)} contracts")
        for label, part in [("no recorded batting/bowling deliveries", idle), (f"under {MIN_DELIVERIES} deliveries", low),
                            (f"overseas, under {MIN_DELIVERIES}", low[low.overseas])]:
            print(f"    {label:40s}{part.price_cr.sum():6.2f} Cr")
        print("    no recorded batting/bowling deliveries: " + ", ".join(f"{p} {c:g} Cr" for p, c in zip(idle.player, idle.price_cr)))
        print(f"    played but not in the file: {', '.join(sorted(on_field - set(s.player))) or 'none'}")
    print("\n  Deliveries means balls faced plus legal balls bowled; fielding-only appearances are not measured.")
    print("  Mid-season replacements are not in the auction file.")
    return rr.sort_values(["year", "price_cr"], ascending=[True, False])[
        ["year", "player", "type", "country", "status", "price_cr", "deliveries", "overseas"]]


# --------------------------------------------------------------------- 2
def squad_check(a, squad):
    """Confirm the squad table's coverage in the auction file, its source for 28 players."""
    rule("2  SQUAD TABLE: coverage and consistency with the auction file")
    country = a.dropna(subset=["country"]).groupby("player").country.last()
    keeper = a.groupby("player").type.agg(lambda t: "WK-Batter" in set(t))
    found = squad[squad.player.isin(keeper.index)]
    missing = squad.player[~squad.player.isin(keeper.index)]
    print(f"  RR squad players in the file: {len(found)} of {len(squad)} "
          f"(missing: {', '.join(missing) or 'none'})")
    print(f"  Nationality mismatches: {int((found.player.map(country) != found.nationality).sum())}")
    print(f"  Keeper mismatches (WK-Batter in either season): "
          f"{int((found.player.map(keeper) != found.keeper).sum())}")


# --------------------------------------------------------------------- 3
def price_profile(a):
    """Add nationality and 2024 contract price to Block 8's scouting profile."""
    rule("3  SCOUTING PROFILE: nationality and 2024 price")
    cand = pd.read_csv(PROC / "rec_scouting_profile.csv")
    country = a.dropna(subset=["country"]).groupby("player").country.last()
    price = a[(a.year == 2024) & a.team.notna()].set_index("player").price_cr
    cand["country"] = cand.bat.map(country)
    cand["price_2024_cr"] = cand.bat.map(price)
    print(f"  {'player':22s} {'country':13s} {'2024 Cr':>8s}")
    for _, r in cand.iterrows():
        nat = r.country if pd.notna(r.country) else "--"
        cost = f"{r.price_2024_cr:.2f}" if pd.notna(r.price_2024_cr) else "--"
        print(f"  {r.bat:22s} {nat:13s} {cost:>8s}")
    ind = cand[cand.country == "India"]
    print(f"\n  {len(ind)} of {len(cand)} are Indian, so the signing needs no overseas slot.")
    print("  2024 contracts are a guide, not 2025 mega-auction prices; retention decides availability.")
    return cand


def main():
    a = load()
    d = pd.read_parquet(PROC / "clean.parquet")
    rr = rr_purse(a, d)
    squad_check(a, pd.read_csv(REF / "rr_squad_reference.csv"))
    cand = price_profile(a)

    rule("WRITING OUTPUT")
    for obj, name in [(rr, "auction_rr_purse.csv"), (cand, "auction_scouting_profile.csv")]:
        obj.to_csv(PROC / name, index=False)
        print(f"  {name:30s} {(PROC / name).stat().st_size / 1024:>7.1f} KB")
    print("\nBlock 10 complete.")


if __name__ == "__main__":
    main()
