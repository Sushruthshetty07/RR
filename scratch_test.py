import pandas as pd
from pathlib import Path

cols = [
    'match_id', 'innings', 'overid', 'over_ball_num',
    'canonical_striker_id', 'canonical_striker_name',
    'runs', 'is_four', 'is_six', 'is_wicket', 'is_batter_ball_faced',
    'pressure_bat'
]
df = pd.read_csv('data/processed/deliveries_with_pressure.csv.gz', usecols=cols)
df = df.sort_values(['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

bf = df[df['is_batter_ball_faced'] == 1].copy()
bf['prev_was_dot'] = (bf.groupby(['match_id', 'innings', 'canonical_striker_id'])['runs'].shift(1) == 0)
bf['prev_pressure'] = bf.groupby(['match_id', 'innings', 'canonical_striker_id'])['pressure_bat'].shift(1)
bf['prev_overid'] = bf.groupby(['match_id', 'innings', 'canonical_striker_id'])['overid'].shift(1)
bf['prev_ball'] = bf.groupby(['match_id', 'innings', 'canonical_striker_id'])['over_ball_num'].shift(1)

jurel_pd = bf[(bf['canonical_striker_id'] == 7260) & (bf['pressure_bat'] >= 60) & (bf['prev_was_dot'] == True) & (bf['prev_pressure'] >= 60)]
print('Jurel post-dot deliveries (Consec-HP):')
for idx, row in jurel_pd.iterrows():
    print(f"Match {row['match_id']} Inn {row['innings']} Over {row['overid']}.{row['over_ball_num']} (Prev: {row['prev_overid']}.{row['prev_ball']}): Runs={row['runs']}, Wkt={row['is_wicket']}, PI={row['pressure_bat']:.1f}")

ashutosh_pd = bf[(bf['canonical_striker_id'] == 8101) & (bf['pressure_bat'] >= 60) & (bf['prev_was_dot'] == True) & (bf['prev_pressure'] >= 60)]
print('\nAshutosh post-dot deliveries with wickets (Consec-HP):')
for idx, row in ashutosh_pd[ashutosh_pd['is_wicket'] == 1].iterrows():
    print(f"Match {row['match_id']} Inn {row['innings']} Over {row['overid']}.{row['over_ball_num']} (Prev: {row['prev_overid']}.{row['prev_ball']}): Runs={row['runs']}, Wkt={row['is_wicket']}, PI={row['pressure_bat']:.1f}")
