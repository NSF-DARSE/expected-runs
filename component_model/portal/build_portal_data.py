"""Build portal buy-low board data (v3).

v2 added control cohorts, survivorship, FB usage %, conference, handedness,
role proxy. v3 (2026-07-23) extends the arsenal to 7 graded types
(+ Sinker/Cutter/Splitter, adopted via extended_types_test.py: paired diff
+0.03 SE 0.03 on 2024->25, +0.04 SE 0.02 on 2025->26) and emits a per-pitcher
`detail` payload for the board tooltip: per pitch type, usage, within-type
Stuff+/Location+ (100-scale), and top-3 physical feature drivers with exact
ridge contributions (context features is_lhp/is_lhb excluded from display).
Output: portal_board.json (aggregates only, no pitch-level data).
"""
import pandas as pd, numpy as np, json

USECOLS = ['Date','Pitcher','PitcherId','PitcherTeam','PitcherThrows','Balls','Strikes',
           'PitchofPA','PitchCall','OutsOnPlay','RunsScored','TaggedPitchType',
           'Target','Level','League']
SWING = {'StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay'}
FF = {'Fastball','FourSeamFastBall'}

def load(path, year):
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    df = df[df.Level == 'D1']
    df['yr'] = pd.to_datetime(df.Date, errors='coerce').dt.year
    df = df[df.yr == year]
    df['PitcherId'] = pd.to_numeric(df.PitcherId, errors='coerce').astype('Int64')
    return df[df.PitcherId.notna()]

def mode_league(s):
    s = s[s != 'Team']
    return s.mode().iat[0] if len(s) else 'Team'

def stats(df):
    d = df.copy()
    d['is_k'] = (d.Strikes == 2) & d.PitchCall.isin(['StrikeSwinging','StrikeCalled'])
    d['is_bb'] = (d.Balls == 3) & (d.PitchCall == 'BallCalled')
    d['is_pa'] = d.PitchofPA == 1
    d['is_swing'] = d.PitchCall.isin(SWING)
    d['is_whiff'] = d.PitchCall == 'StrikeSwinging'
    d['is_fb'] = d.TaggedPitchType.isin(FF)
    d['outs'] = d.OutsOnPlay.fillna(0) + d.is_k.astype(int)
    g = d.groupby('PitcherId').agg(
        pitches=('Target','size'), team=('PitcherTeam', lambda s: s.mode().iat[0]),
        name=('Pitcher', lambda s: s.mode().iat[0]),
        throws=('PitcherThrows', lambda s: s.mode().iat[0]),
        league=('League', mode_league), games=('Date','nunique'),
        runs=('RunsScored','sum'), outs=('outs','sum'),
        k=('is_k','sum'), bb=('is_bb','sum'), pa=('is_pa','sum'),
        swings=('is_swing','sum'), whiffs=('is_whiff','sum'),
        fb=('is_fb','sum'), mean_target=('Target','mean'))
    g['ip'] = g.outs/3
    g['ra9'] = np.where(g.ip > 0, g.runs*9/g.ip, np.nan)
    g['k_pct'] = g.k/g.pa; g['bb_pct'] = g.bb/g.pa
    g['whiff_pct'] = g.whiffs/g.swings
    g['fb_pct'] = g.fb/g.pitches
    g['ppa'] = g.pitches/g.games  # pitches per appearance (role proxy)
    return g

print('loading 2025...')
d25 = load('c:/Users/jackdav/repos/baseball-stuff-plus/Final_Target_Calc_2109.csv', 2025)
print('loading 2026...')
d26 = load('c:/Users/jackdav/repos/baseball-stuff-plus/trackman_api/2026_build/Final_Target_Calc_2026.csv', 2026)
s25, s26 = stats(d25), stats(d26)

# ---- ARSENAL Pitching+ grade (mix-neutral variant B, adopted 2026-07-23) ----
# Per type (FF/SL/CH/CB, script-09 protocol): ridge stuff model + pooled xT
# location map trained on 2025. Quality = prediction minus the type's
# population mean (so pitch MIX is not rewarded, only quality per type).
# Arsenal grade = usage-weighted mean quality across all graded pitches.
import sys as _sys
_sys.path.insert(0, 'c:/Users/jackdav/repos/baseball-stuff-plus/component_model/analysis')
_saved_argv = _sys.argv
_sys.argv = ['x','--data','C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv',
             '--workdir','C:/Users/jackdav/stuffplus_replication/workdir_2526_d1',
             '--years','2025,2026','--level','D1']
import fair_criterion as fc
_args = fc.paths(); _pit = fc.load_pitches(_args); fc.add_xt(_pit)
_sys.argv = _saved_argv
_TYPES = [('FF', None), ('Slider',{'Slider'}), ('ChangeUp',{'ChangeUp'}),
          ('Curveball',{'Curveball'}), ('Sinker',{'Sinker','TwoSeamFastBall'}),
          ('Cutter',{'Cutter'}), ('Splitter',{'Splitter'})]
_parts, _store = [], {}
for _name, _tag in _TYPES:
    _mask = _pit['is_ff'] if _tag is None else _pit['TaggedPitchType'].isin(_tag)
    _pp, _model = fc.stuff_ridge(_pit, pitch_mask=_mask, return_model=True)
    _pp = _pp[_pp['PlateLocSide'].notna() & _pp['PlateLocHeight'].notna()].copy()
    fc.add_loc_bins(_pp)
    _lmap = fc.PooledLocationMap(_pp[(_pp['year']==2024) & _pp['xT'].notna()])
    _pp['loc'] = _lmap.apply(_pp)
    _p1 = _pp[_pp['year']==2024]
    _g = _p1.groupby('PitcherId').agg(n=('ridge_pred','size'),
        stuff=('ridge_pred','mean'), locv=('loc','mean'))
    _g['q_stuff'] = _g.stuff - _p1.ridge_pred.mean()
    _g['q_loc'] = _g.locv - _p1['loc'].mean()
    _parts.append(_g.assign(ptype=_name).reset_index())
    # tooltip detail inputs: pitcher feature means + exact ridge contributions
    _fm = _p1.groupby('PitcherId')[fc.FEATS].mean()
    _sc = _model.named_steps['standardscaler']; _rd = _model.named_steps['ridge']
    _contrib = (_fm - _sc.mean_) / _sc.scale_ * _rd.coef_  # runs, lower = better
    _ref = _g[_g.n >= 50]  # display-scale reference population for this type
    _store[_name] = dict(g=_g, fm=_fm, contrib=_contrib,
        ref_fm=_fm.loc[_ref.index],
        sd_q=_ref.q_stuff.std(), sd_l=_ref.q_loc.std())
_pt = pd.concat(_parts)
_grp = _pt.groupby('PitcherId')
gr = pd.DataFrame({
    'ars_stuff': _grp.apply(lambda x: np.average(x.q_stuff, weights=x.n), include_groups=False),
    'ars_loc':   _grp.apply(lambda x: np.average(x.q_loc, weights=x.n), include_groups=False),
    'n_graded':  _grp.n.sum()})
gr = gr.join(_pt[_pt.ptype=='FF'].set_index('PitcherId').n.rename('n_ff'), how='inner')
gr = gr[gr.n_ff >= 100]
_tot25 = _pit[_pit['year']==2024].groupby('PitcherId').size().rename('n_total')
gr = gr.join(_tot25)
gr['cover_pct'] = gr.n_graded/gr.n_total
def z(s): return (s - s.mean())/s.std()
gr['grade_raw'] = (z(gr.ars_stuff) + z(gr.ars_loc))/2
gr['stuff100'] = 100 - 15*z(gr.ars_stuff)
gr['loc100'] = 100 - 15*z(gr.ars_loc)

m = s25.join(gr, how='inner')
m = m[m.ip >= 20]
m['grade100'] = 100 - 15*(m.grade_raw - m.grade_raw.mean())/m.grade_raw.std()
m['results100'] = 100 - 15*(m.mean_target - m.mean_target.mean())/m.mean_target.std()
m['gap'] = m.grade100 - m.results100

f = m.join(s26, how='inner', lsuffix='_25', rsuffix='_26')
f = f[f.ip_26 >= 15]
f['transfer'] = f.team_25 != f.team_26
print(f'eligible 2025: {len(m)}; with 2026 follow-up: {len(f)}; transfers: {f.transfer.sum()}')

f = f.sort_values('gap', ascending=False)
top, bot = f.head(50), f.tail(50)

def coh(g, sfx=('_25','_26')):
    a, b = sfx
    return dict(n=len(g), ip25=round(g['ip'+a].mean(),1), ip26=round(g['ip'+b].mean(),1),
        ra9_25=round(g['ra9'+a].mean(),2), ra9_26=round(g['ra9'+b].mean(),2),
        k25=round(100*g['k_pct'+a].mean(),1), k26=round(100*g['k_pct'+b].mean(),1),
        bb25=round(100*g['bb_pct'+a].mean(),1), bb26=round(100*g['bb_pct'+b].mean(),1),
        wh25=round(100*g['whiff_pct'+a].mean(),1), wh26=round(100*g['whiff_pct'+b].mean(),1),
        improved_ra9=int((g['ra9'+b] < g['ra9'+a]).sum()))

# --- CONTROL: equally-bad 2025 lines, split by model grade ---
# worst tercile of 2025 results (by mean_target) among the followed-up pool,
# then split by grade tercile within it
f['res_ter'] = pd.qcut(f.mean_target_25, 3, labels=[0,1,2])  # 2 = worst line
f['grade_ter'] = pd.qcut(f.grade_raw, 3, labels=[0,1,2])     # 0 = best grade
badline = f[f.res_ter == 2]
ctl_good = badline[badline.grade_ter == 0]
ctl_bad = badline[badline.grade_ter == 2]
control = dict(good_grade=coh(ctl_good), bad_grade=coh(ctl_bad))
print('CONTROL among worst-line tercile:')
print(' good grade:', control['good_grade'])
print(' bad grade :', control['bad_grade'])

# --- SURVIVORSHIP: attrition into 2026 by cohort, among eligible 2025 pool ---
m['followed'] = m.index.isin(f.index)
m['res_ter_all'] = pd.qcut(m.mean_target, 3, labels=[0,1,2])
m['grade_ter_all'] = pd.qcut(m.grade_raw, 3, labels=[0,1,2])
bl_all = m[m.res_ter_all == 2]
surv = dict(
    pool=dict(n=len(m), followed=int(m.followed.sum()), rate=round(m.followed.mean(),3)),
    badline_goodgrade=dict(n=len(bl_all[bl_all.grade_ter_all==0]),
        followed=int(bl_all[bl_all.grade_ter_all==0].followed.sum()),
        rate=round(bl_all[bl_all.grade_ter_all==0].followed.mean(),3)),
    badline_badgrade=dict(n=len(bl_all[bl_all.grade_ter_all==2]),
        followed=int(bl_all[bl_all.grade_ter_all==2].followed.sum()),
        rate=round(bl_all[bl_all.grade_ter_all==2].followed.mean(),3)))
print('SURVIVORSHIP:', json.dumps(surv))

# --- matched pairs: bad-line group split at grade median, matched on 2025 RA9 ---
badf = f[f.res_ter == 2].copy()
badf['liked_half'] = badf.grade_raw < badf.grade_raw.median()
L = badf[badf.liked_half].sort_values('ra9_25'); D = badf[~badf.liked_half].sort_values('ra9_25')
used, li_idx, di_idx = set(), [], []
for i, r in L.iterrows():
    cand = D[~D.index.isin(used)]
    diffs = (cand.ra9_25 - r.ra9_25).abs()
    if len(diffs) and diffs.min() <= 0.30:
        j = diffs.idxmin(); used.add(j); li_idx.append(i); di_idx.append(j)
li, di = L.loc[li_idx], D.loc[di_idx]
def mcoh(g):
    return dict(ra9_25=round(g.ra9_25.mean(),2), ra9_26=round(g.ra9_26.mean(),2),
        k25=round(100*g.k_pct_25.mean(),1), k26=round(100*g.k_pct_26.mean(),1),
        bb25=round(100*g.bb_pct_25.mean(),1), bb26=round(100*g.bb_pct_26.mean(),1),
        wh25=round(100*g.whiff_pct_25.mean(),1), wh26=round(100*g.whiff_pct_26.mean(),1),
        improved_ra9=int((g.ra9_26 < g.ra9_25).sum()))
matched = dict(n_pairs=len(li), liked=mcoh(li), disliked=mcoh(di),
    note='nearest-neighbor on 2025 RA9, caliper 0.30, worst-line tercile split at grade median')
print('MATCHED (n=%d): liked %s / disliked %s' % (len(li), matched['liked'], matched['disliked']))

# --- regression: grade effect per SD holding the 2025 line fixed ---
rng = np.random.default_rng(7); idx = f.index.values
def boot(cols_fn):
    out = []
    for _ in range(4000):
        s = f.loc[rng.choice(idx, len(idx))]
        X = np.column_stack(cols_fn(s))
        b, *_ = np.linalg.lstsq(X, s.ra9_26, rcond=None)
        out.append(b[-1])
    return np.array(out)
e1 = boot(lambda s: [np.ones(len(s)), s.ra9_25, z(s.grade_raw)])
e2 = boot(lambda s: [np.ones(len(s)), s.ra9_25, 100*s.k_pct_25, 100*s.bb_pct_25, z(s.grade_raw)])
regression = dict(n=len(f),
    effect_per_sd_holding_ra9=round(e1.mean(),2), se=round(e1.std(),2),
    ci=[round(np.percentile(e1,2.5),2), round(np.percentile(e1,97.5),2)],
    effect_per_sd_holding_full_line=round(e2.mean(),2), se_full=round(e2.std(),2),
    ci_full=[round(np.percentile(e2,2.5),2), round(np.percentile(e2,97.5),2)],
    note='next-year RA9 improvement per 1 SD better arsenal Pitching+ grade, bootstrap 4000')
print('REGRESSION:', regression)

summary = dict(liked=coh(top), disliked=coh(bot),
               pool=dict(n=len(f), ra9_25=round(f.ra9_25.mean(),2), ra9_26=round(f.ra9_26.mean(),2)),
               matched=matched, regression=regression, survivorship=surv)

def row(r):
    return dict(name=r['name_25'], id=int(r.name), throws=r['throws_25'],
        team25=r['team_25'], team26=r['team_26'], transfer=bool(r['transfer']),
        lg25=r['league_25'], lg26=r['league_26'],
        fb_pct=round(100*r['fb_pct_25']), cover=round(100*r['cover_pct']), ppa25=round(r['ppa_25']),
        grade=round(r['grade100'],1), stuff=round(r['stuff100'],1), loc=round(r['loc100'],1),
        results=round(r['results100'],1), gap=round(r['gap'],1),
        ip25=round(r['ip_25'],1), ip26=round(r['ip_26'],1),
        ra9_25=round(r['ra9_25'],2), ra9_26=round(r['ra9_26'],2), d_ra9=round(r['ra9_26']-r['ra9_25'],2),
        k25=round(100*r['k_pct_25'],1), k26=round(100*r['k_pct_26'],1), d_k=round(100*(r['k_pct_26']-r['k_pct_25']),1),
        bb25=round(100*r['bb_pct_25'],1), bb26=round(100*r['bb_pct_26'],1), d_bb=round(100*(r['bb_pct_26']-r['bb_pct_25']),1),
        wh25=round(100*r['whiff_pct_25'],1), wh26=round(100*r['whiff_pct_26'],1), d_wh=round(100*(r['whiff_pct_26']-r['whiff_pct_25']),1))

# --- per-pitcher tooltip detail (board rows only) ---
_PHYS = [x for x in fc.FEATS if x not in ('is_lhp','is_lhb')]
_LAB = {'EffectiveVelo':('Velo','mph',1), 'InducedVertBreak':('IVB','in',1),
        'HorzBreak':('Horz Break','in',1), 'SpinRate':('Spin','rpm',0),
        'Extension':('Extension','ft',1), 'RelHeight':('Rel Height','ft',1),
        'RelSide':('Rel Side','ft',1), 'vertbreakdiff':('IVB vs FB','in',1),
        'horzbreakdiff':('HB vs FB','in',1), 'velocity_differential':('Velo vs FB','mph',1)}
def pdetail(pid):
    entries = []
    for _name, _ in _TYPES:
        st = _store[_name]
        if pid not in st['g'].index or st['g'].loc[pid, 'n'] < 10:
            continue
        r = st['g'].loc[pid]
        c = st['contrib'].loc[pid]
        drivers = []
        for ftr in sorted(_PHYS, key=lambda x: -abs(c[x]))[:3]:
            lab, unit, nd = _LAB[ftr]
            raw = st['fm'].loc[pid, ftr]
            drivers.append(dict(f=lab, raw=round(raw, nd), unit=unit,
                pctl=int(round(100*(st['ref_fm'][ftr] < raw).mean())),
                pts=round(-15*c[ftr]/st['sd_q'], 1)))
        entries.append(dict(pt=_name, n=int(r.n),
            stuff=round(100 - 15*r.q_stuff/st['sd_q'], 1),
            loc=round(100 - 15*r.q_loc/st['sd_l'], 1), drivers=drivers))
    entries.sort(key=lambda e: -e['n'])
    tot = sum(e['n'] for e in entries)
    for e in entries:
        e['use'] = round(100*e['n']/tot)
    return entries

_board = pd.concat([f.head(75), f.tail(25)])
detail = {str(int(pid)): pdetail(int(pid)) for pid in _board.index}

out = dict(built='2026-07-23', cohorts=summary,
           top=[row(r) for _, r in f.head(75).iterrows()],
           bottom=[row(r) for _, r in f.tail(25).iterrows()],
           detail=detail)
with open('portal_board.json','w') as fh:
    json.dump(out, fh, indent=1)
print('wrote portal_board.json')
