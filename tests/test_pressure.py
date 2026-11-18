"""
Automated Unit & Integration Tests for src/player_identity.py and src/pressure.py.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets
from src.match_state import reconstruct_match_situations
from src.player_identity import apply_canonical_identity, CANONICAL_ID_MAP
from src.pressure import compute_pressure_scores, compute_player_opportunities, PressureConfig


class TestPlayerIdentity(unittest.TestCase):
    def test_basil_hameed_merge_and_preservation(self):
        df = pd.DataFrame({
            'striker_id': [39444, 8638, 36153, 8729, 7446, 6922, 7338, 28756],
            'striker_name': ['Basil Hameed', 'Basil Hameed', 'Syed Haider', 'Haider Shah',
                             'Abdul Samad', 'Abdul Samad', 'Haider Ali', 'Haider Ali'],
            'non_striker_id': [8638, 39444, 8729, 36153, 6922, 7446, 28756, 7338],
            'non_striker_name': ['Basil Hameed', 'Basil Hameed', 'Haider Shah', 'Syed Haider',
                                 'Abdul Samad', 'Abdul Samad', 'Haider Ali', 'Haider Ali'],
            'bowler_id': [39444, 8638, 36153, 8729, 4722, 20172, 7338, 28756],
            'bowler_name': ['Basil Hameed', 'Basil Hameed', 'Syed Haider', 'Haider Shah',
                            'Rashid Khan', 'Rashid Khan', 'Haider Ali', 'Haider Ali'],
            'dismissed_player_id': [39444.0, np.nan, 36153.0, 8729.0, 7446.0, 6922.0, np.nan, np.nan],
        })
        out = apply_canonical_identity(df)

        # Original IDs preserved verbatim
        self.assertEqual(out['striker_id'].iloc[0], 39444)
        self.assertEqual(out['canonical_striker_id'].iloc[0], 8638)
        self.assertEqual(out['canonical_striker_id'].iloc[1], 8638)
        self.assertEqual(out['canonical_dismissed_player_id'].iloc[0], 8638.0)
        self.assertTrue(pd.isna(out['canonical_dismissed_player_id'].iloc[1]))

        # Kept separate
        self.assertNotEqual(out['canonical_striker_id'].iloc[2], out['canonical_striker_id'].iloc[3])  # 36153 vs 8729
        self.assertNotEqual(out['canonical_striker_id'].iloc[4], out['canonical_striker_id'].iloc[5])  # Abdul Samad
        self.assertNotEqual(out['canonical_striker_id'].iloc[6], out['canonical_striker_id'].iloc[7])  # Haider Ali
        self.assertNotEqual(out['canonical_bowler_id'].iloc[4], out['canonical_bowler_id'].iloc[5])    # Rashid Khan


class TestPressureEngineSynthetic(unittest.TestCase):
    def setUp(self):
        self.comps = pd.DataFrame([
            {'comp_id': 1, 'comp_name': 'IPL 2025', 'match_format': 'T20', 'season': '2025'},
        ])
        self.matches = pd.DataFrame([
            {
                'match_id': 100, 'comp_id': 1, 'revised_target': np.nan,
                'first_inn_revised_overs': np.nan, 'second_inn_revised_overs': np.nan, 'knock_out': 0
            },
            {
                'match_id': 101, 'comp_id': 1, 'revised_target': 90.0,
                'first_inn_revised_overs': np.nan, 'second_inn_revised_overs': 60.0, 'knock_out': 0
            },
        ])
        # Match 100 has Inn 1, Inn 2, plus double Super Over (Inn 3, 4, 5, 6)
        self.bbb = pd.DataFrame([
            {'match_id': 100, 'innings': 1, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 0, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-01 14:00:00 UTC'},
            {'match_id': 100, 'innings': 1, 'overid': 1, 'over_ball_num': 2, 'innings_ball_num': 2,
             'runs': 6, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-01 14:01:00 UTC'},
            {'match_id': 100, 'innings': 2, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 1, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-01 16:00:00 UTC'},
            # Super Over 1 (innings 3 & 4)
            {'match_id': 100, 'innings': 3, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 6, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-01 17:30:00 UTC'},
            {'match_id': 100, 'innings': 4, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 6, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-01 17:35:00 UTC'},
            # Double Super Over (innings 5 & 6)
            {'match_id': 100, 'innings': 5, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 4, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-01 17:45:00 UTC'},
            {'match_id': 100, 'innings': 6, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 4, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-01 17:50:00 UTC'},
            # Match 101 DLS missing timestamp check
            {'match_id': 101, 'innings': 1, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 4, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-02 14:00:00 UTC'},
            {'match_id': 101, 'innings': 2, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'runs': 1, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': np.nan},
        ])

    def test_pre_delivery_outcome_invariance_and_fallback(self):
        state = reconstruct_match_situations(self.bbb, self.matches)
        p1 = compute_pressure_scores(state, self.matches, self.comps)

        # Mutate post-delivery outcome on ball 2 of Inn 1 (last ball of Inn 1) without changing prior state
        bbb_mut = self.bbb.copy()
        bbb_mut['dismissal_type'] = bbb_mut['dismissal_type'].astype(object)
        bbb_mut.loc[1, 'runs'] = 0
        bbb_mut.loc[1, 'is_wicket'] = True
        bbb_mut.loc[1, 'dismissal_type'] = 'Bowled'
        state_mut = reconstruct_match_situations(bbb_mut, self.matches)
        p2 = compute_pressure_scores(state_mut, self.matches, self.comps)

        # Pre-delivery pressure on Inn 1 Ball 2 MUST be identical!
        row_orig = p1[(p1['match_id'] == 100) & (p1['innings'] == 1) & (p1['over_ball_num'] == 2)].iloc[0]
        row_mut = p2[(p2['match_id'] == 100) & (p2['innings'] == 1) & (p2['over_ball_num'] == 2)].iloc[0]
        self.assertAlmostEqual(row_orig['pressure_bat'], row_mut['pressure_bat'])
        self.assertAlmostEqual(row_orig['pressure_bowl'], row_mut['pressure_bowl'])

        # Ball 1 progress = 0 => comp_p_deficit == 0.0 and comp_b_dom == 0.0
        b1 = p1[(p1['match_id'] == 100) & (p1['innings'] == 1) & (p1['over_ball_num'] == 1)].iloc[0]
        self.assertEqual(b1['comp_p_deficit'], 0.0)
        self.assertEqual(b1['comp_b_dom'], 0.0)
        self.assertTrue(b1['loo_low_support_flag'])

    def test_double_super_over_and_ambiguous_dls(self):
        state = reconstruct_match_situations(self.bbb, self.matches)
        p = compute_pressure_scores(state, self.matches, self.comps)

        # Innings 5 & 6 handled separately
        inn6 = p[(p['match_id'] == 100) & (p['innings'] == 6)].iloc[0]
        self.assertTrue(inn6['is_super_over'])
        self.assertEqual(inn6['target'], 5)  # Innings 5 scored 4 => target 5
        self.assertTrue(pd.isna(inn6['pressure_bat']))
        self.assertTrue(85.0 <= inn6['pressure_so_bat'] <= 100.0)

        # Missing timestamp DLS match flagged ambiguous and pressure set to NaN
        m101_i2 = p[(p['match_id'] == 101) & (p['innings'] == 2)].iloc[0]
        self.assertEqual(m101_i2['dls_status'], 'dls_ambiguous_timing')
        self.assertTrue(m101_i2['dls_ambiguous'])
        self.assertTrue(pd.isna(m101_i2['pressure_bat']))
        self.assertTrue(pd.isna(m101_i2['pressure_bowl']))

    def test_synthetic_pre_stoppage_and_opportunity_exclusion(self):
        # Synthetic match with pre-stoppage gap >= 12 min, normal match, and Super Over
        bbb_opp = pd.DataFrame([
            # Match 200: Inn 1 normal (1 ball by batter 10 / bowler 20)
            {'match_id': 200, 'innings': 1, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'striker_id': 10, 'bowler_id': 20, 'runs': 4, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-03 14:00:00 UTC'},
            # Match 200: Super Over Inn 3 (1 ball by batter 10 / bowler 20 - MUST be excluded from reg opportunities)
            {'match_id': 200, 'innings': 3, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'striker_id': 10, 'bowler_id': 20, 'runs': 6, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-03 17:30:00 UTC'},
            # Match 201: DLS Inn 2 pre-stoppage (ball 1, gap 25m before ball 2) -> pre MUST be NaN and excluded!
            {'match_id': 201, 'innings': 1, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'striker_id': 30, 'bowler_id': 40, 'runs': 6, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-04 14:00:00 UTC'},
            {'match_id': 201, 'innings': 2, 'overid': 1, 'over_ball_num': 1, 'innings_ball_num': 1,
             'striker_id': 10, 'bowler_id': 20, 'runs': 1, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-04 16:00:00 UTC'},
            {'match_id': 201, 'innings': 2, 'overid': 1, 'over_ball_num': 2, 'innings_ball_num': 2,
             'striker_id': 10, 'bowler_id': 20, 'runs': 2, 'extras_type': np.nan, 'extras1_runs': 0, 'extras_2_runs': 0,
             'is_wicket': False, 'dismissal_type': np.nan, 'delivery_ist_time': '2025-04-04 16:25:00 UTC'},
        ])
        matches_opp = pd.DataFrame([
            {'match_id': 200, 'comp_id': 1, 'revised_target': np.nan,
             'first_inn_revised_overs': np.nan, 'second_inn_revised_overs': np.nan, 'knock_out': 0},
            {'match_id': 201, 'comp_id': 1, 'revised_target': 5.0,
             'first_inn_revised_overs': np.nan, 'second_inn_revised_overs': 60.0, 'knock_out': 0},
        ])
        st_opp = reconstruct_match_situations(bbb_opp, matches_opp)
        p_opp = compute_pressure_scores(st_opp, matches_opp, self.comps)

        m201_i2 = p_opp[(p_opp['match_id'] == 201) & (p_opp['innings'] == 2)].sort_values('over_ball_num')
        self.assertEqual(m201_i2.iloc[0]['dls_status'], 'dls_pre_stoppage_suspected')
        self.assertTrue(m201_i2.iloc[0]['dls_ambiguous'])
        self.assertTrue(pd.isna(m201_i2.iloc[0]['pressure_bat']))
        self.assertTrue(pd.isna(m201_i2.iloc[0]['pressure_bowl']))

        self.assertEqual(m201_i2.iloc[1]['dls_status'], 'dls_post_stoppage_suspected')
        self.assertFalse(m201_i2.iloc[1]['dls_ambiguous'])
        self.assertTrue(pd.notna(m201_i2.iloc[1]['pressure_bat']))

        bat_opp, bowl_opp = compute_player_opportunities(p_opp)
        # Batter 10 faced: Match 200 Inn1 (1), Match 200 SO (excluded), Match 201 pre-stoppage (excluded), Match 201 post-stoppage (1) => 2 balls
        b10 = bat_opp[bat_opp['striker_id'] == 10].iloc[0]
        w20 = bowl_opp[bowl_opp['bowler_id'] == 20].iloc[0]
        self.assertEqual(b10['total_balls'], 2)
        self.assertEqual(w20['total_balls'], 2)


DATA_RAW_EXISTS = (PROJECT_ROOT / "data" / "raw" / "competitions.csv").exists()


@unittest.skipUnless(DATA_RAW_EXISTS, "Proprietary raw competition dataset not present (kept local per challenge rules)")
class TestPressureEngineFullDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_raw_datasets()
        bbb_id = apply_canonical_identity(cls.ds['ball_by_ball'])
        cls.state = reconstruct_match_situations(bbb_id, cls.ds['matches'])
        cls.df = compute_pressure_scores(
            cls.state, cls.ds['matches'], cls.ds['competitions'], PressureConfig()
        )

    def test_primary_knockout_default_is_one(self):
        self.assertEqual(PressureConfig().knockout_multiplier, 1.00)

    def test_bounds_and_regular_vs_super_over_separation(self):
        reg = self.df[(~self.df['is_super_over']) & (~self.df['dls_ambiguous'])]
        self.assertFalse(reg['pressure_bat'].isna().any())
        self.assertFalse(reg['pressure_bowl'].isna().any())
        self.assertTrue(((reg['pressure_bat'] >= 0.0) & (reg['pressure_bat'] <= 100.0)).all())
        self.assertTrue(((reg['pressure_bowl'] >= 0.0) & (reg['pressure_bowl'] <= 100.0)).all())
        self.assertTrue(reg['pressure_so_bat'].isna().all())

        # ALL ambiguous DLS deliveries must have NaN regular pressure scores
        amb = self.df[self.df['dls_ambiguous']]
        self.assertEqual(len(amb), 143)
        self.assertTrue(amb['pressure_bat'].isna().all())
        self.assertTrue(amb['pressure_bowl'].isna().all())

        so = self.df[self.df['is_super_over']]
        self.assertEqual(len(so), 38)
        self.assertTrue(so['pressure_bat'].isna().all())
        self.assertTrue(so['pressure_bowl'].isna().all())
        self.assertTrue(((so['pressure_so_bat'] >= 85.0) & (so['pressure_so_bat'] <= 100.0)).all())
        self.assertTrue(((so['pressure_so_bowl'] >= 85.0) & (so['pressure_so_bowl'] <= 100.0)).all())

    def test_dls_ipl_2023_final_no_future_leakage(self):
        # Match 21586: rain hit after 3 balls of Inn 2 (gap 140.7m before over 1.4)
        m2 = self.df[(self.df['match_id'] == 21586) & (self.df['innings'] == 2)].sort_values(['overid', 'over_ball_num'])
        pre = m2.iloc[:3]
        post = m2.iloc[3:]
        self.assertTrue((pre['dls_status'] == 'dls_pre_stoppage_suspected').all())
        self.assertTrue((pre['dls_ambiguous']).all())
        self.assertTrue((pre['target_eff'] == 215.0).all())  # NOT future 171!
        self.assertTrue((pre['max_legal_balls_eff'] == 120).all())
        self.assertTrue(pre['pressure_bat'].isna().all())
        self.assertTrue(pre['pressure_bowl'].isna().all())
        self.assertTrue((post['dls_status'] == 'dls_post_stoppage_suspected').all())
        self.assertFalse((post['dls_ambiguous']).any())
        self.assertTrue((post['target_eff'] == 171.0).all())
        self.assertTrue((post['max_legal_balls_eff'] == 90).all())
        self.assertTrue(post['pressure_bat'].notna().all())
        self.assertTrue(post['pressure_bowl'].notna().all())

    def test_confirmed_vs_suspected_dls_separation_and_nan_gating(self):
        pre_susp = self.df[self.df['dls_status'] == 'dls_pre_stoppage_suspected']
        post_susp = self.df[self.df['dls_status'] == 'dls_post_stoppage_suspected']
        break_susp = self.df[self.df['dls_status'] == 'dls_innings_break_suspected']
        conf = self.df[self.df['dls_status'] == 'dls_pre_innings_confirmed']

        # Exact audited counts
        self.assertEqual(len(pre_susp), 143)
        self.assertEqual(len(post_susp), 214)
        self.assertEqual(len(break_susp), 487)
        self.assertEqual(len(conf), 797)

        # Ambiguous pre-stoppage deliveries must NEVER produce regular pressure scores
        self.assertTrue(pre_susp['dls_ambiguous'].all())
        self.assertTrue(pre_susp['pressure_bat'].isna().all())
        self.assertTrue(pre_susp['pressure_bowl'].isna().all())

        # Confirmed and non-ambiguous suspected categories remain disjoint and finite
        self.assertFalse(conf['dls_ambiguous'].any())
        self.assertFalse(break_susp['dls_ambiguous'].any())
        self.assertFalse(post_susp['dls_ambiguous'].any())
        for sub in (conf, break_susp, post_susp):
            self.assertTrue(sub['pressure_bat'].notna().all())
            self.assertTrue(sub['pressure_bowl'].notna().all())

    def test_opportunity_excludes_super_overs_and_ambiguous_dls(self):
        bat_opp, bowl_opp = compute_player_opportunities(self.df)
        expected_bat_balls = int(
            ((~self.df['is_super_over']) & (~self.df['dls_ambiguous']) & self.df['is_batter_ball_faced']).sum()
        )
        expected_bowl_balls = int(
            ((~self.df['is_super_over']) & (~self.df['dls_ambiguous']) & self.df['is_legal_ball']).sum()
        )
        self.assertEqual(int(bat_opp['total_balls'].sum()), expected_bat_balls)
        self.assertEqual(int(bowl_opp['total_balls'].sum()), expected_bowl_balls)

    def test_loo_ball1_and_support_counts_all_indices(self):
        inn1_b1 = self.df[(self.df['innings'] == 1) & (self.df['legal_balls_before'] == 0)]
        self.assertTrue((inn1_b1['comp_p_deficit'] == 0.0).all())
        self.assertTrue((inn1_b1['comp_b_dom'] == 0.0).all())

        # Verify every ball index 0..119 for IPL and T20I separately
        std_inn1 = self.df[(self.df['innings'] == 1) & (self.df['max_legal_balls'] == 120)]
        for is_ipl_flag, min_expected in [(True, 335), (False, 408)]:
            sub = std_inn1[std_inn1['is_ipl'] == is_ipl_flag]
            by_ball = sub.groupby('legal_balls_before')['loo_support_count'].min()
            self.assertEqual(len(by_ball), 120)
            self.assertGreaterEqual(int(by_ball.min()), min_expected)
            self.assertFalse(sub['loo_low_support_flag'].any())

    def test_loo_exclusion_of_self(self):
        # Mutate one IPL match's score at ball 60 and ensure its own LOO baseline is unchanged
        b60 = self.df[(self.df['innings'] == 1) & (self.df['max_legal_balls'] == 120) & (self.df['legal_balls_before'] == 60)]
        sample_mid = b60.iloc[0]['match_id']
        orig_loo = b60[b60['match_id'] == sample_mid].iloc[0]['loo_exp_score']

        bbb_mod = self.ds['ball_by_ball'].copy()
        # Inflate runs in sample_mid Inn 1 first 10 overs
        bbb_mod.loc[(bbb_mod['match_id'] == sample_mid) & (bbb_mod['innings'] == 1) & (bbb_mod['overid'] <= 10), 'runs'] += 10
        st_mod = reconstruct_match_situations(bbb_mod, self.ds['matches'])
        p_mod = compute_pressure_scores(st_mod, self.ds['matches'], self.ds['competitions'], PressureConfig())
        mod_loo = p_mod[(p_mod['match_id'] == sample_mid) & (p_mod['innings'] == 1) & (p_mod['legal_balls_before'] == 60)].iloc[0]['loo_exp_score']
        self.assertAlmostEqual(orig_loo, mod_loo, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
