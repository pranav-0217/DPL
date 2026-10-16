# Cleaning Log

`src/01_clean.py` prints validation counts and writes the processed files.

| Item | Value |
|---|---|
| Input | `Data_Premier_League_Prelims_Dataset.csv`: 34,967 rows × 61 columns |
| Output | `data/processed/clean.parquet`: 34,967 rows × 67 columns |
| Raw data | Unchanged; no rows dropped or imputed |

## Cleaning steps

| Step | Action | Result |
|---|---|---|
| Team names | Replace RCB's `Bangalore` alias with `Bengaluru` | 3,521 cells corrected; RCB's 2023 wins resolve to 7 instead of 0 |
| No result | Replace winner `-` with `NA` and flag match 1359519 | 119 deliveries retained for descriptive analysis |
| Dismissals | Reset 2 retired-hurt events to not out (wicket counter unchanged); exclude run outs, retired out and obstructing from bowler credit | 1,797 dismissals; 1,687 bowler wickets |
| Striker flag | Combine `out` and `bat_out` | Add `striker_dismissed`; never use `bat_out` alone |
| Phase | Map overs 1–6, 7–15 and 16–20 | Add ordered `phase` categories |
| Legal balls | Exclude wides and no-balls | Add `legal_ball`: 33,437 deliveries |
| Shortened matches | Flag every row if any match row has `max_balls != 120` | 639 deliveries across three matches |
| Match table | Aggregate teams, venue, result and flags | `matches.parquet`: 145 rows × 15 columns |
| Run corrections | Reconcile components with cumulative counters | Correct 14 deliveries; restore 32 runs |

## Delivery counts

| Phase | Overs | Deliveries | Share |
|---|---|---|---|
| Powerplay | 1–6 | 10,823 | 31.0% |
| Middle | 7–15 | 15,991 | 45.7% |
| Death | 16–20 | 8,153 | 23.3% |

| Delivery type | Count |
|---|---|
| Wides | 1,355 |
| No-balls | 175 |
| Legal balls | 33,437 |

`ballfaced == 0` identifies wides with zero exceptions in this dataset.

## Wicket attribution

| Raw event | Count |
|---|---|
| Bowler-credited wicket | 1,687 |
| Run out | 107 |
| Retired out | 2 |
| Obstructing the field | 1 |
| **Dismissals** | **1,797** |
| Retired hurt, reset to not out (not a dismissal) | 2 |
| Total flagged in the raw file | 1,799 |

`bat_out` is false on 52 flagged events and true on every delivery without a flagged event.

## Model exclusions

| Match ID | Reason |
|---|---|
| 1359519 | No result; one innings and 119 deliveries |
| 1359476 | Shortened match; includes 93 rows still marked `max_balls = 120` |
| 1370353 | Shortened 2023 final |
| 1426298 | Shortened 2024 match at Eden Gardens |

- `max_balls` counts: 120 on 34,421 rows; 96 on 207; 90 on 94; zero on 245.
- Zero is a source artefact, not a usable innings limit.
- Excluding all four matches leaves 34,209 deliveries from 141 matches.

## Run reconciliation

- Fourteen no-balls with byes or leg-byes contain incorrect scores, including six negative bowler-run values.
- Corrected `score = batruns + wide + noball + byes + legbyes`.
- Corrected `bowlruns = batruns + wide + noball`.
- Each correction must match both cumulative innings and bowler counters; disagreement stops processing.
- `run_corrections.csv` records the 14 original and corrected values.
- `run_reconciliation.csv` confirms exact totals for all 289 innings.

| Runs | Raw | Corrected |
|---|---|---|
| Innings runs | 51,627 | 51,659 |
| Bowler runs | 50,819 | 50,851 |

## Validation

| Check | Result |
|---|---|
| Duplicate delivery IDs / missing key fields | 0 / 0 |
| Run formulas and cumulative counters agree | 100% of processed rows |
| Innings totals reconcile | 289 of 289 |
| Matches / seasons / teams / grounds | 145 / 2 / 10 / 14 |
| Batters / bowlers | 232 / 183 |
| Defended / chased / no result | 75 / 69 / 1 |

## Match coverage

| Team | 2023 | 2024 |
|---|---|---|
| Chennai Super Kings | 16 | 14 |
| Delhi Capitals | 14 | 14 |
| Gujarat Titans | 17 | 12 |
| Kolkata Knight Riders | 14 | 14 |
| Lucknow Super Giants | 15 | 14 |
| Mumbai Indians | 16 | 14 |
| Punjab Kings | 14 | 14 |
| Rajasthan Royals | 14 | 15 |
| Royal Challengers Bengaluru | 14 | 15 |
| Sunrisers Hyderabad | 14 | 16 |

- All 74 matches from 2023 are present, including the four playoffs.
- The 2024 data contains 71 matches; three abandoned games had no deliveries.
- Those missing games involve GT twice, KKR twice, SRH once and RR once.
- The analysis compares scoring through per-ball and per-over rates.

## Auction file

`src/10_auction.py` reads `IPL_Auction_2023_24.xlsx` without changing it; `src/07_squad_reference.py` also reads it for squad nationality and keeper roles.

| Step | Result |
|---|---|
| Prices | Lakh and crore values converted to ₹ crore; every squad row has a price |
| Names | Two spellings aligned to the delivery data |
| Duplicates | Shashank Singh's unsold 2024 row dropped; one row per player and season |
| RR match | 28 of 29 squad-table players found; Keshav Maharaj was a mid-season replacement |
| Squad table | The file supplies nationality and keeper roles for those 28 players |
| RR squad cost | ₹91.65 Cr in 2023 and ₹99.8 Cr in 2024 |

- Zero deliveries means no recorded balls faced or legal balls bowled; it does not establish zero appearances.
- Squad metadata sources are recorded per player; Keshav Maharaj uses [RR's 28 March 2024 announcement](https://www.rajasthanroyals.com/latest-news/keshav-maharaj-replacement-prasidh-krishna-rr-ipl-2024).
- Unverified birth years and age bands are omitted.
