"""Replicate the arsenal-vs-FF grade comparison on the 2024->2025 D1 pair.
Coach stats for both years come from Final_Target_Calc_2109.csv (2024+2025)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'c:/Users/jackdav/repos/baseball-stuff-plus/component_model/analysis')
sys.argv = ['x', '--data', 'c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv',
            '--workdir', 'C:/Users/jackdav/stuffplus_replication/workdir_2425_d1', '--level', 'D1']
import fair_criterion as fc

args = fc.paths()
df = fc.load_pitches(args)   # cache hit: pitches_cache_D1.parquet
fc.add_xt(df)

TYPES = [('FF', None), ('Slider', 'Slider'), ('ChangeUp', 'ChangeUp'), ('Curveball', 'Curveball')]
parts = []
ff_grades = None
for name, tag in TYPES:
    mask = df['is_ff'] if tag is None else (df['TaggedPitchType'] == tag)
    pp = fc.stuff_ridge(df, pitch_mask=mask)
    pp = pp[pp['PlateLocSide'].notna() & pp['PlateLocHeight'].notna()].copy()
    fc.add_loc_bins(pp)
    lmap = fc.PooledLocationMap(pp[(pp['year'] == 2024) & pp['xT'].notna()])
    pp['loc'] = lmap.apply(pp)
    p1 = pp[pp['year'] == 2024][['PitcherId', 'ridge_pred', 'loc']].assign(ptype=name)
    parts.append(p1)
    if name == 'FF':
        ff_grades = p1.groupby('PitcherId').agg(n_ff=('ridge_pred','size'),
            ff_stuff=('ridge_pred','mean'), ff_loc=('loc','mean'))
    print(f'{name}: graded 2024 pitches = {len(p1):,}')

ars = pd.concat(parts).groupby('PitcherId').agg(ars_stuff=('ridge_pred','mean'), ars_loc=('loc','mean'))

# coach stats from the shared build helpers (defs only, no loads)
src = open('C:/Users/jackdav/stuffplus_replication/build_portal_data.py').read()
exec(src.split("print('loading 2025...')")[0])
d24 = load('c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv', 2024)
d25 = load('c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv', 2025)
s1, s2 = stats(d24), stats(d25)

m = s1.join(ff_grades[ff_grades.n_ff >= 100], how='inner').join(ars, how='inner')
m = m[m.ip >= 20]
pool = m.join(s2[['ip','ra9','k_pct','bb_pct']], how='inner', lsuffix='_1', rsuffix='_2')
pool = pool[pool.ip_2 >= 15].dropna(subset=['ra9_1','ra9_2'])
print(f'2024->2025 D1 pool: {len(pool)}')

def z(s): return (s - s.mean())/s.std()
pool['g_ff'] = (z(pool.ff_stuff) + z(pool.ff_loc))/2
pool['g_ars'] = (z(pool.ars_stuff) + z(pool.ars_loc))/2

rng = np.random.default_rng(7)
idx = pool.index.values
def boot(cols):
    out = []
    for _ in range(4000):
        s = pool.loc[rng.choice(idx, len(idx))]
        X = np.column_stack([np.ones(len(s)), s.ra9_1, 100*s.k_pct_1, 100*s.bb_pct_1] +
                            [z(s[c]) for c in cols])
        b, *_ = np.linalg.lstsq(X, s.ra9_2, rcond=None)
        out.append(b[4:])
    return np.array(out)

for cols, lbl in [(['g_ff'],'FF-only'), (['g_ars'],'arsenal')]:
    e = boot(cols)[:,0]
    print(f'{lbl:<8} effect per SD (holding RA9+K+BB): {e.mean():+.2f}  SE={e.std():.2f}  CI=[{np.percentile(e,2.5):+.2f},{np.percentile(e,97.5):+.2f}]')
j = boot(['g_ff','g_ars'])
print(f'joint: FF {j[:,0].mean():+.2f} (SE {j[:,0].std():.2f}), arsenal {j[:,1].mean():+.2f} (SE {j[:,1].std():.2f})')
print(f'corr(g_ff, g_ars) = {pool.g_ff.corr(pool.g_ars):.3f}')
