"""Does grading Sinker/Cutter/Splitter (on top of FF/SL/CH/CB) improve the
arsenal grade's buy-low effect?

Same screening protocol as arsenal_weighting_test.py: mix-neutral variant B,
equal stuff/loc weights, effect = next-year RA9 per SD of grade holding
RA9+K%+BB% fixed, both D1 pairs, paired bootstrap on the difference.
Decision rule (pre-stated): adopt the 7-type grade if it is >= the 4-type
grade minus 1 SE on the 2025->2026 pair and not worse on 2024->2025.
TwoSeamFastBall is merged into Sinker (same pitch family). Sweeper excluded
(zero 2024 pitches -> no training data in the 2425 pair).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'c:/Users/jackdav/repos/baseball-stuff-plus/component_model/analysis')

PAIRS = {
 '2425': dict(argv=['x','--data','c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv',
                    '--workdir','C:/Users/jackdav/stuffplus_replication/workdir_2425_d1','--level','D1'],
              y1=2024, y2=2025,
              src1='c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv',
              src2='c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv'),
 '2526': dict(argv=['x','--data','C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv',
                    '--workdir','C:/Users/jackdav/stuffplus_replication/workdir_2526_d1',
                    '--years','2025,2026','--level','D1'],
              y1=2025, y2=2026,
              src1='c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv',
              src2='c:/Users/jackdav/repos/baseball-stuff-plus/trackman_api/2026_build/Final_Target_Calc_2026.csv'),
}
BASE = [('FF', None), ('Slider',{'Slider'}), ('ChangeUp',{'ChangeUp'}), ('Curveball',{'Curveball'})]
EXT = BASE + [('Sinker',{'Sinker','TwoSeamFastBall'}), ('Cutter',{'Cutter'}), ('Splitter',{'Splitter'})]

helpers = open('C:/Users/jackdav/stuffplus_replication/build_portal_data.py').read().split("print('loading 2025...')")[0]
exec(helpers)  # load(), stats()

def per_type_table(pair, types):
    import importlib, fair_criterion as fc
    sys.argv = pair['argv']
    importlib.reload(fc)
    args = fc.paths(); df = fc.load_pitches(args); fc.add_xt(df)
    rows = []
    for name, tags in types:
        mask = df['is_ff'] if tags is None else df['TaggedPitchType'].isin(tags)
        pp = fc.stuff_ridge(df, pitch_mask=mask)
        pp = pp[pp['PlateLocSide'].notna() & pp['PlateLocHeight'].notna()].copy()
        fc.add_loc_bins(pp)
        lmap = fc.PooledLocationMap(pp[(pp['year']==2024) & pp['xT'].notna()])
        pp['loc'] = lmap.apply(pp)
        p1 = pp[pp['year']==2024]
        g = p1.groupby('PitcherId').agg(n=('ridge_pred','size'),
            stuff=('ridge_pred','mean'), locv=('loc','mean'))
        g['q_stuff'] = g.stuff - p1.ridge_pred.mean()
        g['q_loc'] = g.locv - p1['loc'].mean()
        rows.append(g.assign(ptype=name).reset_index())
        print(f'  {name}: {int(g.n.sum())} pitches, {len(g)} pitchers (year 1)')
    return pd.concat(rows)

def arsenal(pt, names):
    sub = pt[pt.ptype.isin(names)]
    g = sub.groupby('PitcherId')
    return pd.DataFrame({
        'stuff': g.apply(lambda x: np.average(x.q_stuff, weights=x.n), include_groups=False),
        'loc':   g.apply(lambda x: np.average(x.q_loc, weights=x.n), include_groups=False)})

def z(s): return (s - s.mean())/s.std()

def effect(p, grade, seed=7):
    rng = np.random.default_rng(seed); idx = p.index.values; out = []
    for _ in range(4000):
        s = p.loc[rng.choice(idx, len(idx))]
        X = np.column_stack([np.ones(len(s)), s.ra9_1, 100*s.k_pct_1, 100*s.bb_pct_1, z(grade.loc[s.index])])
        b, *_ = np.linalg.lstsq(X, s.ra9_2, rcond=None)
        out.append(b[4])
    e = np.array(out)
    return e.mean(), e.std()

def paired_diff(p, g_ext, g_base, seed=7):
    """bootstrap distribution of effect(ext) - effect(base) on shared resamples"""
    rng = np.random.default_rng(seed); idx = p.index.values; out = []
    for _ in range(4000):
        s = p.loc[rng.choice(idx, len(idx))]
        base_X = [np.ones(len(s)), s.ra9_1, 100*s.k_pct_1, 100*s.bb_pct_1]
        be, *_ = np.linalg.lstsq(np.column_stack(base_X+[z(g_ext.loc[s.index])]), s.ra9_2, rcond=None)
        bb_, *_ = np.linalg.lstsq(np.column_stack(base_X+[z(g_base.loc[s.index])]), s.ra9_2, rcond=None)
        out.append(be[4] - bb_[4])
    d = np.array(out)
    return d.mean(), d.std()

for key, pair in PAIRS.items():
    print(f'=== {key} ===')
    pt = per_type_table(pair, EXT)
    d1 = load(pair['src1'], pair['y1']); d2 = load(pair['src2'], pair['y2'])
    s1, s2 = stats(d1), stats(d2)
    a4 = arsenal(pt, [n for n,_ in BASE])
    a7 = arsenal(pt, [n for n,_ in EXT])
    ff_n = pt[pt.ptype=='FF'].set_index('PitcherId').n.rename('n_ff')
    m = s1.join(a4.add_suffix('4'), how='inner').join(a7.add_suffix('7')).join(ff_n)
    m = m[(m.n_ff >= 100) & (m.ip >= 20)]
    p = m.join(s2[['ip','ra9','k_pct','bb_pct']], how='inner', lsuffix='_1', rsuffix='_2')
    p = p[p.ip_2 >= 15].dropna(subset=['ra9_1','ra9_2'])
    print(f'pool n={len(p)}')
    # coverage improvement
    tot = pt.groupby('PitcherId').n.sum()
    cov4 = pt[pt.ptype.isin([n for n,_ in BASE])].groupby('PitcherId').n.sum()
    cov = pd.DataFrame({'c4': cov4, 'c7': tot}).loc[p.index]
    n_all = None  # coverage vs all pitches needs totals; report graded-pitch counts instead
    print(f'graded pitches per pitcher: 4-type median {cov.c4.median():.0f}, 7-type median {cov.c7.median():.0f} '
          f'(+{(cov.c7/cov.c4-1).median()*100:.0f}% median)')
    g4 = z(p.stuff4) + z(p['loc4'])
    g7 = z(p.stuff7) + z(p['loc7'])
    m4, s4_ = effect(p, g4); m7, s7_ = effect(p, g7)
    dm, ds = paired_diff(p, g7, g4)
    print(f'  4-type effect per SD = {m4:+.2f} SE={s4_:.2f}')
    print(f'  7-type effect per SD = {m7:+.2f} SE={s7_:.2f}')
    print(f'  paired diff (7-4)    = {dm:+.2f} SE={ds:.2f}')
    print(f'  corr(g4, g7) = {g4.corr(g7):.3f}')
