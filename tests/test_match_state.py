"""
Automated Validation & Unit Test Suite for src/match_state.py
and Raw Competition Dataset Integrity.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets
from src.match_state import (
    reconstruct_match_situations,
    BOWLER_WICKET_TYPES,
    NON_LEGAL_EXTRAS,
)


class TestMatchStateSynthetic(unittest.TestCase):
    """Deterministic unit tests verifying pre-delivery causal math."""

    def setUp(self):
        self.matches = pd.DataFrame([
            {
                'match_id': 1,
                'revised_target': np.nan,
                'first_inn_revised_overs': np.nan,
                'second_inn_revised_overs': np.nan,
                'knock_out': 0,
            },
            {
                'match_id': 2,
                'revised_target': 50.0,
                'first_inn_revised_overs': 30.0,  # 5 overs = 30 balls
                'second_inn_revised_overs': np.nan,
                'knock_out': 1,
            }
        ])

        # Synthetic balls for match 1 (inn 1 & 2) and match 2 (DLS) and super over
        self.bbb = pd.DataFrame([
            # Match 1 Inn 1: Ball 1 is wide (1 extra), Ball 2 is 4 runs, Ball 3 is Overstep + 6, Ball 4 is RunOut
            {'match_id': 1, 'innings': 1, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 0, 'extras_type': 'wide', 'extras1_runs': 1, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan},
            {'match_id': 1, 'innings': 1, 'overid': 1, 'over_ball_num': 2, 'innings_ball_num': 2,
             'runs': 4, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan},
            {'match_id': 1, 'innings': 1, 'overid': 1, 'over_ball_num': 3, 'innings_ball_num': 3,
             'runs': 6, 'extras_type': 'Overstep', 'extras1_runs': 1, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan},
            {'match_id': 1, 'innings': 1, 'overid': 1, 'over_ball_num': 4, 'innings_ball_num': 4,
             'runs': 1, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': True, 'dismissal_type': 'RunOut'},
            # Match 1 Inn 2: Target should be 13 + 1 = 14
            {'match_id': 1, 'innings': 2, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 2, 'extras_type': 'legbyes', 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': True, 'dismissal_type': 'Bowled'},
            {'match_id': 1, 'innings': 2, 'overid': 16, 'over_ball_num': 1, 'innings_ball_num': 2,
             'runs': 0, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan},
            # Match 1 Super Over Inn 3 & 4
            {'match_id': 1, 'innings': 3, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 6, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan},
            {'match_id': 1, 'innings': 4, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 4, 'extras_type': 'other', 'extras1_runs': 1, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan},
            # Match 2 DLS Inn 2 (second_inn_revised_overs is NaN, inherits 30 balls from first_inn_revised_overs)
            {'match_id': 2, 'innings': 2, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 0, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan},
        ])

    def test_pre_delivery_state_and_extras(self):
        out = reconstruct_match_situations(self.bbb, self.matches)

        # Ball 1 of Match 1 Inn 1 (wide): state BEFORE must be 0/0, 120 balls rem, CRR 0.0
        b1 = out.iloc[0]
        self.assertEqual(b1['score_before'], 0)
        self.assertEqual(b1['wickets_before'], 0)
        self.assertEqual(b1['legal_balls_before'], 0)
        self.assertEqual(b1['legal_balls_remaining_before'], 120)
        self.assertEqual(b1['crr_before'], 0.0)
        self.assertFalse(b1['is_legal_ball'])
        self.assertFalse(b1['is_batter_ball_faced'])
        self.assertEqual(b1['bowler_runs_conceded'], 1)

        # Ball 2 (after wide): score_before=1, legal_balls_before=0
        b2 = out.iloc[1]
        self.assertEqual(b2['score_before'], 1)
        self.assertEqual(b2['legal_balls_before'], 0)
        self.assertEqual(b2['legal_balls_remaining_before'], 120)
        self.assertTrue(b2['is_legal_ball'])
        self.assertTrue(b2['is_batter_ball_faced'])

        # Ball 3 (Overstep + 6): score_before=5, legal_balls_before=1, CRR = 5 / (1/6) = 30.0
        b3 = out.iloc[2]
        self.assertEqual(b3['score_before'], 5)
        self.assertEqual(b3['legal_balls_before'], 1)
        self.assertAlmostEqual(b3['crr_before'], 30.0)
        self.assertFalse(b3['is_legal_ball'])
        self.assertTrue(b3['is_batter_ball_faced'])
        self.assertEqual(b3['bowler_runs_conceded'], 7)

        # Ball 4 (RunOut): score_before=12, wickets_before=0, legal_balls_before=1
        b4 = out.iloc[3]
        self.assertEqual(b4['score_before'], 12)
        self.assertEqual(b4['wickets_before'], 0)
        self.assertEqual(b4['wickets_in_hand_before'], 10)
        self.assertFalse(b4['is_bowler_wicket'])  # RunOut is NOT bowler wicket

    def test_chase_target_rrr_and_dls_and_super_over(self):
        out = reconstruct_match_situations(self.bbb, self.matches)

        # Match 1 Inn 2 Ball 1: target = 13 + 1 = 14, runs_needed=14, balls_rem=120 => RRR=0.7
        m1_i2_b1 = out[(out['match_id'] == 1) & (out['innings'] == 2)].iloc[0]
        self.assertEqual(m1_i2_b1['target'], 14)
        self.assertEqual(m1_i2_b1['runs_needed_before'], 14)
        self.assertAlmostEqual(m1_i2_b1['rrr_before'], 14 / 20.0)
        self.assertTrue(m1_i2_b1['is_bowler_wicket'])  # Bowled
        self.assertEqual(m1_i2_b1['match_phase'], 'Powerplay')

        # Match 1 Inn 2 Ball 2 (over 16): wickets_before=1, score_before=2, balls_rem=119
        m1_i2_b2 = out[(out['match_id'] == 1) & (out['innings'] == 2)].iloc[1]
        self.assertEqual(m1_i2_b2['wickets_before'], 1)
        self.assertEqual(m1_i2_b2['wickets_in_hand_before'], 9)
        self.assertEqual(m1_i2_b2['runs_needed_before'], 12)
        self.assertEqual(m1_i2_b2['legal_balls_remaining_before'], 119)
        self.assertEqual(m1_i2_b2['match_phase'], 'Death')

        # Match 1 Super Over Inn 4: target = 6 + 1 = 7, max_legal_balls=6, wickets_in_hand=2, phase='Super Over'
        m1_so = out[(out['match_id'] == 1) & (out['innings'] == 4)].iloc[0]
        self.assertEqual(m1_so['target'], 7)
        self.assertEqual(m1_so['max_legal_balls'], 6)
        self.assertEqual(m1_so['wickets_in_hand_before'], 2)
        self.assertEqual(m1_so['match_phase'], 'Super Over')
        self.assertFalse(m1_so['is_legal_ball'])  # 'other' is non-legal

        # Match 2 DLS Inn 2: inherits 30 balls from first_inn_revised_overs, target = 50
        m2_dls = out[out['match_id'] == 2].iloc[0]
        self.assertEqual(m2_dls['max_legal_balls'], 30)
        self.assertEqual(m2_dls['legal_balls_remaining_before'], 30)
        self.assertEqual(m2_dls['target'], 50)
        self.assertAlmostEqual(m2_dls['rrr_before'], 10.0)

    def test_bowler_runs_conceded_all_extras_categories(self):
        """Verify conc_i charges wide/Overstep/other extras1_runs + bat runs, while excluding byes/legbyes/Penalty and extras_2_runs."""
        matches = pd.DataFrame([
            {'match_id': 99, 'revised_target': np.nan, 'first_inn_revised_overs': np.nan, 'second_inn_revised_overs': np.nan, 'knock_out': 0}
        ])
        bbb = pd.DataFrame([
            # 1. Normal scoring delivery (4 off bat)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 4, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0, 'is_wicket': False, 'dismissal_type': np.nan},
            # 2. Wide + 4 wides (5 extras1_runs -> all 5 charged)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 2, 'innings_ball_num': 2,
             'runs': 0, 'extras_type': 'wide', 'extras1_runs': 5, 'extras_2_runs': 0, 'is_wicket': False, 'dismissal_type': np.nan},
            # 3. Overstep no-ball + 6 off bat (1 extra1 + 6 bat -> 7 charged)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 3, 'innings_ball_num': 3,
             'runs': 6, 'extras_type': 'Overstep', 'extras1_runs': 1, 'extras_2_runs': 0, 'is_wicket': False, 'dismissal_type': np.nan},
            # 4. Overstep no-ball + 4 secondary byes/legbyes (extras_2_runs=4 -> only 1 charged to bowler, 5 total team runs)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 4, 'innings_ball_num': 4,
             'runs': 0, 'extras_type': 'Overstep', 'extras1_runs': 1, 'extras_2_runs': 4, 'is_wicket': False, 'dismissal_type': np.nan},
            # 5. 'other' no-ball (e.g. above waist) + 1 secondary legbye + 2 off bat (3 charged, 4 total)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 5, 'innings_ball_num': 5,
             'runs': 2, 'extras_type': 'other', 'extras1_runs': 1, 'extras_2_runs': 1, 'is_wicket': False, 'dismissal_type': np.nan},
            # 6. Byes (4 extras1_runs -> 0 charged, legal ball)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 6, 'innings_ball_num': 6,
             'runs': 0, 'extras_type': 'byes', 'extras1_runs': 4, 'extras_2_runs': 0, 'is_wicket': False, 'dismissal_type': np.nan},
            # 7. Legbyes (2 extras1_runs -> 0 charged, legal ball)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 7, 'innings_ball_num': 7,
             'runs': 0, 'extras_type': 'legbyes', 'extras1_runs': 2, 'extras_2_runs': 0, 'is_wicket': False, 'dismissal_type': np.nan},
            # 8. Penalty runs (5 extras1_runs -> 0 charged)
            {'match_id': 99, 'innings': 1, 'overid': 1, 'over_ball_num': 8, 'innings_ball_num': 8,
             'runs': 0, 'extras_type': 'Penalty', 'extras1_runs': 5, 'extras_2_runs': 0, 'is_wicket': False, 'dismissal_type': np.nan},
        ])
        out = reconstruct_match_situations(bbb, matches)
        expected_conc = [4, 5, 7, 1, 3, 0, 0, 0]
        expected_total = [4, 5, 7, 5, 4, 4, 2, 5]
        expected_legal = [True, False, False, False, False, True, True, True]
        expected_faced = [True, False, True, True, True, True, True, True]
        self.assertEqual(out['bowler_runs_conceded'].tolist(), expected_conc)
        self.assertEqual(out['ball_total_runs'].tolist(), expected_total)
        self.assertEqual(out['is_legal_ball'].tolist(), expected_legal)
        self.assertEqual(out['is_batter_ball_faced'].tolist(), expected_faced)


class TestFullDatasetIntegration(unittest.TestCase):
    """Integration tests on the full 200,999-delivery competition dataset."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = load_raw_datasets()
        cls.matches = cls.datasets['matches']
        cls.bbb = cls.datasets['ball_by_ball']
        cls.df = reconstruct_match_situations(cls.bbb, cls.matches)

    def test_row_preservation_and_no_duplicates(self):
        self.assertEqual(len(self.df), len(self.bbb))
        self.assertEqual(len(self.df), 200999)
        self.assertEqual(
            self.df.duplicated(subset=['match_id', 'innings', 'overid', 'over_ball_num']).sum(),
            0
        )

    def test_first_ball_of_every_innings_is_zero_state(self):
        first_balls = self.df.groupby(['match_id', 'innings']).first()
        self.assertTrue((first_balls['score_before'] == 0).all())
        self.assertTrue((first_balls['wickets_before'] == 0).all())
        self.assertTrue((first_balls['legal_balls_before'] == 0).all())
        self.assertTrue((first_balls['crr_before'] == 0.0).all())

    def test_legal_balls_within_bounds(self):
        self.assertGreaterEqual(self.df['legal_balls_remaining_before'].min(), 1)
        self.assertTrue((self.df['legal_balls_before'] < self.df['max_legal_balls']).all())

    def test_ipl_2023_final_dls(self):
        # Match 21586: CSK vs GT IPL 2023 Final, 15 overs (90 balls), target 171
        m2 = self.df[(self.df['match_id'] == 21586) & (self.df['innings'] == 2)]
        self.assertTrue((m2['max_legal_balls'] == 90).all())
        self.assertTrue((m2['target'] == 171).all())
        self.assertEqual(m2.iloc[0]['legal_balls_remaining_before'], 90)
        self.assertAlmostEqual(m2.iloc[0]['rrr_before'], 171 / 15.0)

    def test_shortened_match_inherited_overs(self):
        # Match 16753: first_inn_revised_overs = 66, second_inn_revised_overs = NaN
        m2 = self.df[(self.df['match_id'] == 16753) & (self.df['innings'] == 2)]
        self.assertTrue((m2['max_legal_balls'] == 66).all())

    def test_super_over_invariants(self):
        so = self.df[self.df['innings'] >= 3]
        self.assertEqual(len(so), 38)
        self.assertTrue((so['match_phase'] == 'Super Over').all())
        self.assertTrue((so['max_legal_balls'] == 6).all())
        self.assertTrue(so['wickets_in_hand_before'].isin([1, 2]).all())

    def test_wicket_attribution_counts(self):
        self.assertEqual(self.df['is_wicket'].sum(), 10881)
        self.assertEqual(self.df['is_bowler_wicket'].sum(), 10272)
        self.assertEqual((self.df['is_wicket'] & ~self.df['is_bowler_wicket']).sum(), 609)

    def test_overid_and_phases(self):
        reg = self.df[self.df['innings'] <= 2]
        self.assertEqual(reg['overid'].min(), 1)
        self.assertEqual(reg['overid'].max(), 20)
        self.assertTrue(set(reg['match_phase'].unique()) == {'Powerplay', 'Middle', 'Death'})

    def test_bowler_runs_conceded_extras_invariants(self):
        # Byes & Penalty never charge bowler
        self.assertEqual(self.df.loc[self.df['extras_type'] == 'byes', 'bowler_runs_conceded'].sum(), 0)
        self.assertEqual(self.df.loc[self.df['extras_type'] == 'Penalty', 'bowler_runs_conceded'].sum(), 0)
        # Wide & Overstep/other charge extras1_runs + runs, excluding secondary extras_2_runs
        for et in ['wide', 'Overstep', 'other']:
            sub = self.df[self.df['extras_type'] == et]
            np.testing.assert_array_equal(sub['bowler_runs_conceded'].values, (sub['runs'] + sub['extras1_runs']).values)
        # Verify secondary byes/legbyes off no-balls (83 runs) are not charged to bowler
        nb_sec = self.df[self.df['extras_type'].isin(['Overstep', 'other']) & (self.df['extras_2_runs'] > 0)]
        self.assertEqual(int(nb_sec['extras_2_runs'].sum()), 83)
        self.assertTrue((nb_sec['ball_total_runs'] - nb_sec['bowler_runs_conceded'] == nb_sec['extras_2_runs']).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
