"""Does grading the whole arsenal (FF + SL + CH + CB) improve the buy-low
signal over the FF-only Pitching+ grade?

Per type (script-09 protocol): Ridge stuff model trained on the 2025 season
(train role), pooled xT location map trained same year; per-pitch predicted
run values are unit-consistent across types, so the arsenal grade is the
pitcher's mean predicted value over all graded pitches (usage-weighted by
construction). Test: next-year RA9 regression holding the 2025 surface line
(RA9, K%, BB%) fixed - FF-only grade vs arsenal grade.
Sign convention: predictions are expected runs, lower = better.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'c:/Users/jackdav/repos/baseball-stuff-plus/component_model/analysis')
sys.argv = ['x', '--data', 'C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv',
            '--workdir', 'C:/Users/jackdav/stuffplus_replication/workdir_2526_d1',
            '--years', '2025,2026', '--level', 'D1']
import fair_criterion as fc

args = fc.paths()
df = fc.load_pitches(args)
fc.add_xt(df)

TYPES = [('FF', None), ('Slider', 'Slider'), ('ChangeUp', 'ChangeUp'), ('Curveball', 'Curveball')]
parts = []
for name, tag in TYPES:
    mask = df['is_ff'] if tag is None else (df['TaggedPitchType'] == tag)
    pp = fc.stuff_ridge(df, pitch_mask=mask)
    pp = pp[pp['PlateLocSide'].notna() & pp['PlateLocHeight'].notna()].copy()
    fc.add_loc_bins(pp)
    lmap = fc.PooledLocationMap(pp[(pp['year'] == 2024) & pp['xT'].notna()])
    pp['loc'] = lmap.apply(pp)
    p1 = pp[pp['year'] == 2024][['PitcherId', 'ridge_pred', 'loc']].assign(ptype=name)
    parts.append(p1)
    print(f'{name}: graded 2025 pitches = {len(p1):,}')

allp = pd.concat(parts)
ars = allp.groupby('PitcherId').agg(n_graded=('ridge_pred','size'),
    ars_stuff=('ridge_pred','mean'), ars_loc=('loc','mean'))

# total 2025 pitches per pitcher for coverage
tot = df[df['year'] == 2024].groupby('PitcherId').size().rename('n_total')
ars = ars.join(tot)
ars['coverage'] = ars.n_graded / ars.n_total

# board pool: rebuild the 543-pitcher frame (FF grade + coach stats)
sys.argv = ['x']
exec(open('C:/Users/jackdav/stuffplus_replication/build_portal_data.py').read()
     .split("f = f.sort_values('gap', ascending=False)")[0])
pool = f.join(ars, how='inner')
print(f'pool with arsenal grade: {len(pool)}; median coverage of arsenal: {pool.coverage.median():.0%}')

def z(s): return (s - s.mean())/s.std()
pool['g_ff'] = (z(pool.stuff_raw) + z(pool.loc_raw))/2          # FF-only Pitching+
pool['g_ars'] = (z(pool.ars_stuff) + z(pool.ars_loc))/2        # arsenal Pitching+

rng = np.random.default_rng(7)
idx = pool.index.values
def boot_effect(gcol):
    out = []
    for _ in range(4000):
        s = pool.loc[rng.choice(idx, len(idx))]
        X = np.column_stack([np.ones(len(s)), s.ra9_25, 100*s.k_pct_25, 100*s.bb_pct_25, z(s[gcol])])
        b, *_ = np.linalg.lstsq(X, s.ra9_26, rcond=None)
        out.append(b[4])
    return np.array(out)

for gcol, lbl in [('g_ff','FF-only'), ('g_ars','arsenal')]:
    e = boot_effect(gcol)
    print(f'{lbl:<8} grade effect per SD (holding RA9+K+BB): {e.mean():+.2f}  SE={e.std():.2f}  95% CI=[{np.percentile(e,2.5):+.2f},{np.percentile(e,97.5):+.2f}]')

# head-to-head: both grades in one regression - does arsenal add beyond FF?
diffs = []
for _ in range(4000):
    s = pool.loc[rng.choice(idx, len(idx))]
    X = np.column_stack([np.ones(len(s)), s.ra9_25, 100*s.k_pct_25, 100*s.bb_pct_25, z(s.g_ff), z(s.g_ars)])
    b, *_ = np.linalg.lstsq(X, s.ra9_26, rcond=None)
    diffs.append((b[4], b[5]))
diffs = np.array(diffs)
print(f'joint model: FF coef {diffs[:,0].mean():+.2f} (SE {diffs[:,0].std():.2f}), arsenal coef {diffs[:,1].mean():+.2f} (SE {diffs[:,1].std():.2f})')
print(f'corr(g_ff, g_ars) = {pool.g_ff.corr(pool.g_ars):.3f}')
