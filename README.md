# IPL Strategic Review — Rajasthan Royals

Data Premier League 2026 · Preliminary Round · Question 1

Scouting, performance and strategy analysis of Rajasthan Royals using IPL 2023–24 data.

## Reproduce

Use Python 3.12 and place the organisers' two files in the repository root: `Data_Premier_League_Prelims_Dataset.csv` and `IPL_Auction_2023_24.xlsx`.

```bash
pip install -r requirements.txt
python src/01_clean.py               # clean and validate
python src/02_benchmark.py           # compare teams and phases
python src/03_figures.py             # draw core charts
python src/04_diagnosis.py           # examine RR's middle overs
python src/05_model.py               # compare win-probability models
python src/06_context.py             # examine venue, opponent and shot context
python src/07_squad_reference.py     # build the squad table
python src/08_recommendations.py     # review roles and scouting profiles
python src/09_deck_figures.py        # draw presentation charts
python src/10_auction.py             # price RR's contracts and scouting profile
python -m unittest discover -s tests # run regression tests
```

Scripts print validation results and use random seed 42.

## Files

| Path | Purpose |
|---|---|
| `src/` | Analysis scripts, run in order from 01 to 10 |
| `tests/` | Regression tests for cleaning and model exclusions |
| `data/processed/` | Generated result tables |
| `data/reference/` | Squad metadata and derived player summaries |
| `docs/` | Data dictionary and cleaning log |
| `reports/figures/` | Analysis and presentation charts |
| `deck/` | Final presentation |

## Main results

- Middle-overs edge has the highest correlation with season win percentage: 0.558 (p = 0.010), versus powerplay 0.486 and death 0.343.
- Both champions ranked first in middle-overs edge; RR's 2023 middle overs ranked bottom-half on all seven quality metrics.
- RR's 2024 middle-overs control was second-highest at 80.1% (p = 0.027 vs the league), with 1.53 runs per controlled ball (p = 0.042); KKR recorded 71.4% and 1.87.
- RR's attacking-shot share (sweeps, pulls, hooks, slogs, ramps, upper cuts and reverses) was lowest at 18.2%, but its attacking strike rate was highest at 229.
- Against full balls, RR trailed the league by 28 strike-rate points outside off and 24 at the stumps.
- The illustrative shot-mix scenario raises RR's strike rate from 132.5 to 137.5; fixed shot returns make this estimate optimistic.
- RR used 12 overseas players across two seasons; Boult and Buttler accounted for 55% of the overseas players' batting and bowling deliveries.
- Prasidh Krishna was retained at ₹10 Cr in both seasons and bowled no balls.
- In 2024, players with no recorded batting or bowling deliveries accounted for ₹14.5 Cr of RR's ₹99.8 Cr in contracts.
- Overseas players with under 60 deliveries in a season cost ₹2.25 Cr in 2023 and ₹2.4 Cr in 2024, so the churn was cheap.
- 13 of the 20 players with the profile RR lack are Indian, so the signing needs no overseas slot.

## Figures

All charts are in `reports/figures/`; only the two `deck_` charts appear on the slides.

| Figure | What it shows |
|---|---|
| `deck_s2_framing.png` | Season win % for every team, with RR and each champion highlighted (slide 2) |
| `deck_s4_overview.png` | RR against each season's champion: league rank by phase for batting strike rate and bowling economy (slide 4) |
| `fig1_phase_vs_success.png` | How closely each phase's net edge tracks season win %, with 95% intervals, and how widely teams differ in each phase |
| `fig2_champion_validation.png` | Middle-overs net edge against season win % for all 20 team-seasons; both champions led the league |
| `fig3_rr_rank_heatmap.png` | RR's league rank on seven quality metrics by season and phase; the 2023 middle overs are bottom-half on all seven |
| `fig4_uncertainty.png` | RR's strike-rate gap to the other nine teams by phase and season, with bootstrap intervals; none is significant |
| `fig5_model_calibration.png` | Calibration of the win-probability model against the supplied `wprob`, and log-loss for four models |
| `fig6_phase_swing.png` | Win-probability swing per ball by phase; middle-overs deliveries move the result least |
| `fig7_pitch_map.png` | RR's middle-overs strike rate against the league by line and length; weakest against full balls |
| `fig8_shot_and_venue.png` | Middle-overs attacking-shot share by team, and RR's strike-rate gap to the league at each ground |
| `fig9_player_intent.png` | Attacking-shot share against shrunk middle-overs strike rate for each batter, with RR's highlighted |

`src/03_figures.py` draws figures 1–4, `src/05_model.py` 5–6, `src/06_context.py` 7–8, `src/08_recommendations.py` 9 and `src/09_deck_figures.py` the two slide charts.

## Model

Predict the final result from each delivery's match state, using cross-validation grouped by match.

| Model | Log-loss | Brier | AUC |
|---|---|---|---|
| Supplied `wprob` | 0.5721 | 0.1945 | 0.7821 |
| Gradient boosting | 0.5824 | 0.2005 | 0.7469 |
| Logistic regression | 0.5572 | 0.1908 | 0.7787 |
| Logistic regression + derived features | 0.5515 | 0.1888 | 0.7824 |

- Evaluation uses 34,209 deliveries from 141 matches; `wprob` and `predscore` are excluded from model inputs.
- The log-loss improvement is not statistically significant: +0.021, with a 95% interval of −0.014 to +0.058.
- Configurations share scoring folds, so the selected model's score may be optimistic.

## Limits

- Two seasons provide limited evidence; no single-season RR phase strike-rate gap is significant.
- Neither the spin weakness (p = 0.569) nor the late-2024 death collapse (corrected p = 0.118) is supported.
- "Passive" and reading control as risk appetite are interpretations of the shot and control tags, not direct measures of intent.
- Venue and opponent samples are small, and shot-mix results are scenarios rather than forecasts.
- Fielding and injury records are absent; auction prices are 2023–24 contracts, not 2025 mega-auction values, and mid-season replacements are not in the auction file.

## Data and sources

- The organisers' dataset supplies all match and delivery data; the raw CSV is unchanged.
- The organisers' auction file supplies each squad's 2023 and 2024 prices, nationality and roles; it is read without changes.
- Cleaning handles team aliases, wicket attribution, legal balls, incomplete matches and 14 scoring errors; see [Cleaning Log](docs/CLEANING_LOG.md).
- [Data Dictionary](docs/DATA_DICTIONARY.md) defines all 61 source columns, six derived columns and the auction file.
- Squad nationality and keeper roles come from the auction file for 28 players; each row records its source.
- Keshav Maharaj's details come from [RR's announcement, 28 March 2024](https://www.rajasthanroyals.com/latest-news/keshav-maharaj-replacement-prasidh-krishna-rr-ipl-2024); unverified ages are omitted.
- No scraped data is used; the organisers' files are supplied separately.
