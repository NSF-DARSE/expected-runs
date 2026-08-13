"""Should the arsenal grade weight Stuff and Location equally?

Sweep w in grade = w*z(ars_stuff) + (1-w)*z(ars_loc) (mix-neutral variant B
components) and evaluate the buy-low effect (next-year RA9 per SD of grade,
holding RA9+K%+BB% fixed) on both D1 pairs. Also: joint regression with both
components free (the empirically ideal ratio), fit on 2425 only, tested on
2526 per screening discipline.
"""
import sys, numpy as np, pandas as pd
exec(open('C:/Users/jackdav/stuffplus_replication/arsenal_weighting_test.py').read()
     .split("# learned weights")[0])
# `results` now holds the 2425/2526 pools with B_stuff/B_loc and coach stats

WS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
for key, p in results.items():
    print(f'--- {key} (n={len(p)}) ---')
    for w in WS:
        grade = w*z(p.B_stuff) + (1-w)*z(p.B_loc)
        m_, s_ = effect(p, grade)
        print(f'  w_stuff={w:.2f}: effect per SD = {m_:+.2f}  SE={s_:.2f}')
    # joint: both components free
    rng = np.random.default_rng(7); idx = p.index.values; out = []
    for _ in range(4000):
        s = p.loc[rng.choice(idx, len(idx))]
        X = np.column_stack([np.ones(len(s)), s.ra9_1, 100*s.k_pct_1, 100*s.bb_pct_1,
                             z(s.B_stuff), z(s.B_loc)])
        b, *_ = np.linalg.lstsq(X, s.ra9_2, rcond=None)
        out.append((b[4], b[5]))
    out = np.array(out)
    print(f'  joint coefs: stuff {out[:,0].mean():+.2f} (SE {out[:,0].std():.2f}), '
          f'loc {out[:,1].mean():+.2f} (SE {out[:,1].std():.2f})')
    print(f'  corr(z_stuff, z_loc) = {p.B_stuff.corr(p.B_loc):.3f}')

# out-of-sample check of the 2425-implied ratio applied to 2526
tr = results['2425']
Xtr = np.column_stack([np.ones(len(tr)), tr.ra9_1, 100*tr.k_pct_1, 100*tr.bb_pct_1,
                       z(tr.B_stuff), z(tr.B_loc)])
bt, *_ = np.linalg.lstsq(Xtr, tr.ra9_2, rcond=None)
ws, wl = bt[4], bt[5]
w_fit = ws/(ws+wl)
print(f'2425-fitted stuff share = {w_fit:.2f}')
p = results['2526']
m_, s_ = effect(p, w_fit*z(p.B_stuff) + (1-w_fit)*z(p.B_loc))
print(f'applied OOS to 2526: effect per SD = {m_:+.2f}  SE={s_:.2f}')
