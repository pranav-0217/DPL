"""Regression tests for cleaning; run with: python -m unittest discover -s tests"""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cleaning", ROOT / "src/01_clean.py")
cleaning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaning)
sys.path.insert(0, str(ROOT / 'src'))
model_spec = importlib.util.spec_from_file_location("model", ROOT / "src/05_model.py")
model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)
auction_spec = importlib.util.spec_from_file_location("auction", ROOT / "src/10_auction.py")
auction = importlib.util.module_from_spec(auction_spec)
auction_spec.loader.exec_module(auction)


class RunCorrectionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "Data_Premier_League_Prelims_Dataset.csv", low_memory=False)

    def test_corrected_runs_match_independent_running_counters(self):
        df = self.raw.sample(frac=1, random_state=42).copy()
        original_order = df.row_id.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            audit = cleaning.fix_delivery_runs(df)
            reconciliation = cleaning.reconcile_runs(df)
        self.assertEqual(len(audit), 14)
        self.assertEqual(int((df.score - self.raw.score).sum()), 32)
        self.assertEqual(int((df.bowlruns - self.raw.bowlruns).sum()), 32)
        self.assertTrue(df.bowlruns.ge(0).all())
        self.assertEqual(len(reconciliation), 289)
        self.assertTrue(reconciliation['diff'].eq(0).all())
        self.assertEqual(int(df.score.sum()), 51659)
        pd.testing.assert_series_equal(df.row_id, original_order)
        unchanged = self.raw.columns.difference(['score', 'bowlruns'])
        pd.testing.assert_frame_equal(df[unchanged].sort_index(), self.raw[unchanged])

        ordered = df.sort_values(['p_match', 'inns', 'ball_id'])
        innings_delta = ordered.groupby(['p_match', 'inns']).inns_runs.diff().fillna(ordered.inns_runs)
        bowler_delta = (ordered.groupby(['p_match', 'inns', 'p_bowl']).cur_bowl_runs.diff()
                        .fillna(ordered.cur_bowl_runs))
        self.assertTrue(ordered.score.eq(innings_delta).all())
        self.assertTrue(ordered.bowlruns.eq(bowler_delta).all())

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(cleaning.fix_delivery_runs(df).empty)

    def test_conflicting_source_evidence_stops_correction(self):
        df = self.raw.copy()
        df.loc[0, 'inns_runs'] += 1
        before = df.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'cumulative'):
                cleaning.fix_delivery_runs(df)
        pd.testing.assert_frame_equal(df, before)

    def test_shortened_flag_covers_every_delivery_of_affected_matches(self):
        df = self.raw.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            cleaning.flag_shortened(df)
        affected_ids = {1359476, 1370353, 1426298}
        self.assertEqual(set(df.loc[df.shortened, 'p_match']), affected_ids)
        self.assertEqual(int(df.shortened.sum()), 639)
        self.assertTrue(df.groupby('p_match').shortened.nunique().eq(1).all())
        self.assertTrue(df.loc[df.p_match == 1359476, 'shortened'].all())
        self.assertEqual(int((df.shortened & df.max_balls.eq(120)).sum()), 93)
        self.assertFalse(df.loc[~df.p_match.isin(affected_ids), 'shortened'].any())

    def test_model_excludes_whole_matches_even_with_legacy_flags(self):
        df = self.raw.copy()
        # Deliberately reproduce the old mixed flag to test the model's boundary.
        df['shortened'] = df.max_balls.ne(120)
        df['no_result'] = df.winner.eq('-')
        with contextlib.redirect_stdout(io.StringIO()):
            built = model.build(df)
        excluded = {1359476, 1370353, 1426298, 1359519}
        self.assertTrue(set(built.p_match).isdisjoint(excluded))
        self.assertEqual(built.p_match.nunique(), 141)
        self.assertEqual(len(built), 34209)
        # All retained matches keep their complete set of original deliveries.
        pd.testing.assert_series_equal(built.groupby('p_match').size(),
                                       df[df.p_match.isin(built.p_match)].groupby('p_match').size())


class AuctionFileTest(unittest.TestCase):
    def test_prices_names_and_duplicates(self):
        with contextlib.redirect_stdout(io.StringIO()):
            a = auction.load()
        self.assertEqual(auction.crore('20.00 L'), 0.2)
        self.assertEqual(auction.crore('2.00 Cr'), 2.0)
        self.assertTrue(pd.isna(auction.crore('--')))
        self.assertFalse(a.duplicated(['year', 'player']).any())
        self.assertFalse(set(auction.NAME_FIXES) & set(a.player))
        self.assertTrue(a.loc[a.team.notna(), 'price_cr'].notna().all())
        purse = a[a.team == 'Rajasthan Royals'].groupby('year').price_cr.sum().round(2)
        self.assertEqual(purse.to_dict(), {2023: 91.65, 2024: 99.8})


if __name__ == '__main__':
    unittest.main()
