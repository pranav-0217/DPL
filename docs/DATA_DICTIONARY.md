# Data Dictionary

- Source: the organisers' `Data_Premier_League_Prelims_Dataset.csv`.
- Coverage: IPL 2023–24; 34,967 deliveries, 145 matches, 10 teams, 14 grounds, 232 batters and 183 bowlers.
- Processed file: `data/processed/clean.parquet`; 61 source columns and six derived columns.
- Definitions were checked or inferred from the data because no dictionary was supplied.

| Status | Meaning |
|---|---|
| Verified | Supported by a data check |
| Inferred | Supported interpretation with remaining uncertainty |
| Unresolved | Meaning unknown; use conservatively |

## Identifiers

| Column | Type | Status | Definition |
|---|---|---|---|
| `row_id` | int | Verified | Unique delivery ID; no duplicates |
| `p_match` | int | Verified | Match ID; 145 values in ESPNcricinfo's 2023–24 ID ranges |
| `p_bat` | int | Verified | Striker ID; 232 players |
| `p_bowl` | int | Verified | Bowler ID; 183 players |
| `p_out` | int | Verified | ID of the player recorded as out or retired |
| `ball_id` | float | Verified | Over-and-delivery identifier; `0.01` is the first delivery |

## Teams and match context

| Column | Type | Status | Definition |
|---|---|---|---|
| `inns` | int | Verified | Innings number: 1 or 2 |
| `bat` / `bowl` | str | Verified | Striker / bowler name |
| `team_bat` / `team_bowl` | str | Verified | Batting / bowling team; RCB uses `Bengaluru` |
| `ball` | int | Verified | Delivery number within the over, including extra deliveries: 1–8 |
| `over_num` | int | Verified | Over number: 1–20 |
| `match_date` | str | Verified | ISO date: 2023-03-31 to 2024-05-26 |
| `year` | int | Verified | Season: 2023 or 2024 |
| `ground` | str | Verified | Venue; 14 grounds in India |
| `country` | str | Verified | Always `India` |
| `competition` | str | Verified | Always `IPL` |
| `winner` | str | Verified | Winner; RCB aliases normalised and no-result `-` replaced with `NA` |
| `toss` | str | Verified | Toss winner; same RCB alias correction as `winner` |
| `max_balls` | int | Inferred | Innings limit: usually 120, sometimes 90/96; zero is a source artefact |

## Ball outcome

| Column | Type | Status | Definition |
|---|---|---|---|
| `outcome` | str | Verified | No run, run, four, six, out, wide, no ball, leg bye or bye |
| `score` | int | Verified | Total runs: `batruns + wide + noball + byes + legbyes` |
| `batruns` | int | Verified | Batter runs: 0–6, excluding extras |
| `bowlruns` | int | Verified | Bowler runs: `batruns + wide + noball`, excluding byes and leg-byes |
| `ballfaced` | int | Verified | Batter's balls-faced flag; zero only on wides |
| `wide` | int | Verified | Wide runs: 0–5 |
| `noball` | int | Verified | No-ball penalty: 0 or 1 |
| `byes` / `legbyes` | int | Verified | Bye / leg-bye runs |

- Fourteen no-ball corrections restore 32 runs; all 289 innings reconcile with the running totals.
- Audit files: `data/processed/run_corrections.csv` and `data/processed/run_reconciliation.csv`.

## Dismissal fields

| Column | Type | Status | Definition |
|---|---|---|---|
| `out` | bool | Verified | Dismissal flag; 1,799 in the raw file, 1,797 after 2 retired-hurt events are reset (wicket counter unchanged) |
| `dismissal` | str | Verified | Event type on every raw flagged event, including the 2 retired-hurt rows reset to not out |
| `bat_out` | bool | Verified | Striker flag; interpret only with `out`, since it is also true on non-event deliveries |

Event counts: caught 1,270; bowled 268; lbw 115; run out 107; stumped 33; retired out 2; hit wicket 1; obstructing 1; plus 2 retired-hurt events, which are not dismissals.

## Running match state

Counters include the current delivery, checked against the opening over of match 1359475.

| Column | Type | Status | Definition |
|---|---|---|---|
| `cur_bat_runs` / `cur_bat_bf` | int | Verified | Striker's runs / balls faced so far |
| `cur_bowl_ovr` / `cur_bowl_wkts` / `cur_bowl_runs` | float/int | Verified | Bowler's overs / wickets / runs so far |
| `inns_runs` / `inns_wkts` / `inns_balls` | int | Verified | Innings runs / wickets / legal balls so far |
| `inns_rr` | float | Verified | Current run rate |
| `inns_runs_rem` | float | Verified | Runs required in the chase; populated on 48% of rows |
| `inns_balls_rem` | int | Verified | Scheduled balls remaining |
| `inns_rrr` | float | Verified | Required run rate; second innings only |
| `target` | float | Verified | Runs needed to win: 90–288; second innings only |

## Bowler attributes

| Column | Type | Status | Definition |
|---|---|---|---|
| `bowl_kind` | str | Verified | Pace, spin or mixture-unknown |
| `bowl_style` | str | Verified | Arm and bowling style; see notation below |
| `bowl_type` | str | Verified | Delivery type, such as outswinger or googly; 31 categories, 94% filled |
| `bowl_speed_category` | float | Inferred | Ordinal band 1–8, higher meaning slower; 93% filled, with unknown km/h boundaries |
| `bowl_release_point` | float | Unresolved | Per-delivery category 1–4; use unlabelled, not as over/round the wicket |

- Style notation: R/L = arm; F/FM/MF/M = pace; OB = off-break; LB/LBG = leg-spin; SLA = left-arm orthodox; LWS = left-arm wrist-spin.
- Speed-band evidence: RF averages 2.46, RFM 2.72, OB 5.22 and LWS 5.66.
- Release-point evidence: 165 of 183 bowlers use multiple values; categories 1–2 cover about 97% of recorded values.
- Release point is 93% filled; its exact meaning is unconfirmed and does not clearly distinguish over/round the wicket.

## Delivery and shot tags

| Column | Type | Status | Definition |
|---|---|---|---|
| `line` | str | Verified | WIDE_DOWN_LEG, DOWN_LEG, ON_THE_STUMPS, OUTSIDE_OFFSTUMP or WIDE_OUTSIDE_OFFSTUMP |
| `length` | str | Verified | YORKER, FULL_TOSS, FULL, GOOD_LENGTH, SHORT_OF_A_GOOD_LENGTH or SHORT |
| `shot` | str | Verified | Shot category; 25 values, 99% filled |
| `bat_hand` | str | Verified | LHB or RHB |
| `control` | float | Verified | 1 = controlled/intended shot; 0 = beaten, edged or mistimed; 97% filled, mean about 0.73 |
| `wagonx` / `wagony` | int | Verified | Landing coordinates, 0–363; centre about (180, 180), both zero when no placement is recorded |
| `wagonzone` | int | Verified | Ground-fixed scoring sector 0–8; zero means no recorded placement |

## Wagon zones

Zone labels use right-handed shots; percentages show each named shot's share in that zone.

| Zone | Region | Evidence |
|---|---|---|
| 1 | Fine leg | LEG_GLANCE 62%, HOOK 41% |
| 2 | Square leg | SWEEP 57%, LEG_GLANCE 36% |
| 3 | Midwicket | FLICK 61%, SLOG_SWEEP 61%, PULL 52% |
| 4 | Long on | STRAIGHT_DRIVE 52%, ON_DRIVE 43% |
| 5 | Long off | ON_DRIVE 48%, STRAIGHT_DRIVE 46% |
| 6 | Cover | COVER_DRIVE 84% |
| 7 | Point | SQUARE_DRIVE 62%, LATE_CUT 55%, CUT 53% |
| 8 | Third man | UPPER_CUT 59%, REVERSE_SWEEP 45%, LATE_CUT 43% |

- Flip left-handed zones before combining batters: `1↔8`, `2↔7`, `3↔6`, `4↔5`.
- Mirroring evidence: mean runs peak in zones 1–2 for right-handers and 7–8 for left-handers.
- Zone zero is scoreless on 99.6% of its deliveries and is dominated by `LEFT_ALONE`.

## Supplied model outputs

These provider-generated predictions are excluded from model inputs.

| Column | Type | Status | Definition |
|---|---|---|---|
| `predscore` | int | Verified | Live projected first-innings-equivalent total |
| `wprob` | float | Verified | Batting team's win probability, on a 0–100 scale |

- `predscore`: final-ball correlation with the innings total is 1.000; the six-over prediction has MAE 21.3 runs and correlation 0.65.
- `wprob`: final completed-chase means are 100.0 for wins and 2.6 for losses; the loss median is zero.

## Derived columns

`src/01_clean.py` adds these six columns.

| Column | Type | Definition |
|---|---|---|
| `phase` | category | Powerplay 1–6, Middle 7–15, Death 16–20; delivery shares 31.0%, 45.7%, 23.3% |
| `bowler_wicket` | bool | `out` excluding run outs, retirements and obstruction; 1,687 events |
| `striker_dismissed` | bool | `out AND bat_out` |
| `legal_ball` | bool | Neither wide nor no-ball; 33,437 deliveries |
| `no_result` | bool | Match 1359519; 119 deliveries |
| `shortened` | bool | Any row in the match has `max_balls != 120`; 639 deliveries across three matches |

## Auction file

The organisers' `IPL_Auction_2023_24.xlsx` has one sheet per season and one row per squad or unsold player; `src/07_squad_reference.py` and `src/10_auction.py` read it without changes.

| Column | Type | Definition |
|---|---|---|
| `Player Name` | str | Player; two spellings aligned to the delivery data: Donavon → Donovan Ferreira, Philip → Phil Salt |
| `Type` | str | Batter, Bowler, Allrounder or WK-Batter |
| `Country` | str | Nationality; blank for seven overseas players, none of them RR players or scouting candidates |
| `Status` | str | Retained, Sold, Traded or Unsold |
| `Base Price` / `Final Price` | str | Lakh (`L`) or crore (`Cr`), converted to ₹ crore; `--` when there is no price |
| `Team` | str | Franchise code, or `Unsold` |

- Squads are recorded at auction time, so mid-season replacements such as Sandeep Sharma (2023), Keshav Maharaj and Tanush Kotian (2024) are absent.
- Shashank Singh appears twice in 2024, as sold and unsold; the sold row is kept.

## Squad reference

- `metadata_source` cites each player's nationality and keeper-role source; unverified birth years and ages are omitted.
- The auction file covers 28 players; Keshav Maharaj uses [RR's 28 March 2024 announcement](https://www.rajasthanroyals.com/latest-news/keshav-maharaj-replacement-prasidh-krishna-rr-ipl-2024).

## Coverage limits

- Three abandoned 2024 matches contain no deliveries and are absent from the data.
- Match 1359519 has one innings, 119 deliveries and no result.
- Age and batting-order position are absent; nationality, prices and squads come from the auction file.
- Fielding events and Impact Player substitutions are not recorded.
