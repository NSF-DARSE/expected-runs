"""Which arsenal-grade weighting predicts next-year RA9 best (holding the
surface line fixed)?

Variants (grades in run units, lower = better):
  A usage    - mean predicted value over all graded pitches (current; embeds pitch mix)
  B mixneut  - usage-weighted mean of (prediction - pitch-type population mean):
               pure quality-vs-type-average, mix removed
  C shrunk   - B with per-type empirical shrinkage n/(n+51) toward type average
  D learned  - per-type quality z-scores with weights FIT ON 2024->2025 ONLY,
               applied out-of-sample to 2025->2026
Decision rule (pre-stated): if variants are within ~1 SE of each other on the
2025->2026 effect, keep the simplest (A).
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
TYPES = [('FF', None), ('Slider','Slider'), ('ChangeUp','ChangeUp'), ('Curveball','Curveball')]
N0 = 51

helpers = open('C:/Users/jackdav/stuffplus_replication/build_portal_data.py').read().split("print('loading 2025...')")[0]
exec(helpers)  # load(), stats()

def per_type_table(pair):
    """Per-pitcher per-type (n, stuff quality, loc quality) + type means, year-1."""
    import importlib, fair_criterion as fc
    sys.argv = pair['argv']
    importlib.reload(fc)
    args = fc.paths(); df = fc.load_pitches(args); fc.add_xt(df)
    rows, tmeans = [], {}
    for name, tag in TYPES:
        mask = df['is_ff'] if tag is None else (df['TaggedPitchType'] == tag)
        pp = fc.stuff_ridge(df, pitch_mask=mask)
        pp = pp[pp['PlateLocSide'].notna() & pp['PlateLocHeight'].notna()].copy()
        fc.add_loc_bins(pp)
        lmap = fc.PooledLocationMap(pp[(pp['year']==2024) & pp['xT'].notna()])
        pp['loc'] = lmap.apply(pp)
        p1 = pp[pp['year']==2024]
        tmeans[name] = dict(stuff=p1.ridge_pred.mean(), loc=p1['loc'].mean())
        g = p1.groupby('PitcherId').agg(n=('ridge_pred','size'),
            stuff=('ridge_pred','mean'), loc=('loc','mean')).assign(ptype=name)
        rows.append(g.reset_index())
    return pd.concat(rows), tmeans

def pools(pair):
    d1 = load(pair['src1'], pair['y1']); d2 = load(pair['src2'], pair['y2'])
    s1, s2 = stats(d1), stats(d2)
    return s1, s2

results = {}
for key, pair in PAIRS.items():
    pt, tm = per_type_table(pair)
    s1, s2 = pools(pair)
    # quality = prediction minus type population mean
    pt['q_stuff'] = pt.apply(lambda r: r.stuff - tm[r.ptype]['stuff'], axis=1)
    pt['q_loc'] = pt.apply(lambda r: r['loc'] - tm[r.ptype]['loc'], axis=1)
    pt['shr'] = pt.n/(pt.n + N0)
    g = pt.groupby('PitcherId')
    agg = pd.DataFrame({
        'A_stuff': g.apply(lambda x: np.average(x.stuff, weights=x.n), include_groups=False),
        'A_loc':   g.apply(lambda x: np.average(x['loc'], weights=x.n), include_groups=False),
        'B_stuff': g.apply(lambda x: np.average(x.q_stuff, weights=x.n), include_groups=False),
        'B_loc':   g.apply(lambda x: np.average(x.q_loc, weights=x.n), include_groups=False),
        'C_stuff': g.apply(lambda x: np.average(x.q_stuff*x.shr, weights=x.n), include_groups=False),
        'C_loc':   g.apply(lambda x: np.average(x.q_loc*x.shr, weights=x.n), include_groups=False),
    })
    ff_n = pt[pt.ptype=='FF'].set_index('PitcherId').n.rename('n_ff')
    # per-type quality z columns for the learned variant (missing type -> 0)
    wide = pt.pivot_table(index='PitcherId', columns='ptype', values='q_stuff')
    widel = pt.pivot_table(index='PitcherId', columns='ptype', values='q_loc')
    for t,_ in TYPES:
        agg[f'qs_{t}'] = ((wide[t]-wide[t].mean())/wide[t].std()).fillna(0)
        agg[f'ql_{t}'] = ((widel[t]-widel[t].mean())/widel[t].std()).fillna(0)
    m = s1.join(agg, how='inner').join(ff_n, how='inner')
    m = m[(m.n_ff >= 100) & (m.ip >= 20)]
    p = m.join(s2[['ip','ra9','k_pct','bb_pct']], how='inner', lsuffix='_1', rsuffix='_2')
    p = p[p.ip_2 >= 15].dropna(subset=['ra9_1','ra9_2'])
    results[key] = p
    print(f'{key}: pool n={len(p)}')

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

# learned weights: fit on 2425 (RA9_2 on line + 8 per-type quality z's)
tr = results['2425']
Xtr = np.column_stack([np.ones(len(tr)), tr.ra9_1, 100*tr.k_pct_1, 100*tr.bb_pct_1] +
                      [tr[f'qs_{t}'] for t,_ in TYPES] + [tr[f'ql_{t}'] for t,_ in TYPES])
bt, *_ = np.linalg.lstsq(Xtr, tr.ra9_2, rcond=None)
w = bt[4:]
print('learned weights (2425): stuff', dict(zip([t for t,_ in TYPES], w[:4].round(2))),
      'loc', dict(zip([t for t,_ in TYPES], w[4:].round(2))))

for key, p in results.items():
    print(f'--- {key} ---')
    for v in ['A','B','C']:
        grade = z(p[f'{v}_stuff']) + z(p[f'{v}_loc'])
        m_, s_ = effect(p, grade)
        print(f'  {v}: effect per SD = {m_:+.2f}  SE={s_:.2f}')
    if key == '2526':
        learned = sum(w[i]*p[f'qs_{t}'] for i,(t,_) in enumerate(TYPES)) + \
                  sum(w[4+i]*p[f'ql_{t}'] for i,(t,_) in enumerate(TYPES))
        m_, s_ = effect(p, learned)
        print(f'  D (weights fit on 2425, out-of-sample): effect per SD = {m_:+.2f}  SE={s_:.2f}')
