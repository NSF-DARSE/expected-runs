"""Waste-type decomposition + stability of individual location features.

Q1: Is waste rate the better coach target? Does the KIND of waste matter
    (horizontal misses vs low vs high)?
Q2: Year-over-year reliability + predictive validity of each individual
    location feature inside Location+.

Sign convention: adjT is expected runs from the pitcher's perspective,
LOWER = better. A trait with POSITIVE corr vs next-year mean adjT predicts
WORSE outcomes. Zone bands copied from script 02:
  heart  |x|<=0.558, 1.83<=z<=3.17
  shadow |x|<=1.108, 1.17<=z<=3.83
  chase  |x|<=1.658, 0.50<=z<=4.50
  waste  everything else
Waste subtype (hierarchical): low (z<0.5), high (z>4.5), horiz (|x|>1.658, z in band).
"""
import pandas as pd, numpy as np

MIN_P = 100  # min FF pitches per pitcher-year

def features(g):
    x, z = g.PlateLocSide.values, g.PlateLocHeight.values
    ax = np.abs(x)
    heart = (ax <= 0.558) & (z >= 1.83) & (z <= 3.17)
    shadow = (ax <= 1.108) & (z >= 1.17) & (z <= 3.83) & ~heart
    chase = (ax <= 1.658) & (z >= 0.5) & (z <= 4.5) & ~heart & ~shadow
    waste = ~(heart | shadow | chase | ~np.isfinite(x) | ~np.isfinite(z))
    low = waste & (z < 0.5)
    high = waste & (z > 4.5)
    horiz = waste & ~low & ~high
    return pd.Series({
        'n': len(g), 'adjT': g.adjT.mean(),
        'heart_pct': heart.mean(), 'shadow_pct': shadow.mean(),
        'chase_pct': chase.mean(), 'waste_pct': waste.mean(),
        'waste_low': low.mean(), 'waste_high': high.mean(), 'waste_horiz': horiz.mean(),
        'sd_x': np.nanstd(x), 'sd_z': np.nanstd(z), 'mean_z': np.nanmean(z),
        'adjT_low': g.adjT.values[low].mean() if low.sum() >= 5 else np.nan,
        'adjT_high': g.adjT.values[high].mean() if high.sum() >= 5 else np.nan,
        'adjT_horiz': g.adjT.values[horiz].mean() if horiz.sum() >= 5 else np.nan,
        'adjT_waste': g.adjT.values[waste].mean() if waste.sum() >= 5 else np.nan,
        'adjT_nonwaste': g.adjT.values[~waste].mean(),
    })

FEATS = ['heart_pct','shadow_pct','chase_pct','waste_pct',
         'waste_low','waste_high','waste_horiz','sd_x','sd_z','mean_z']

for label, path in [('2024->2025 D1', 'workdir_2425_d1/ff_panel_D1.parquet'),
                    ('2025->2026 D1', 'workdir_2526_d1/ff_panel_2025_2026_D1.parquet')]:
    df = pd.read_parquet(path)
    df = df[df.PlateLocSide.notna() & df.PlateLocHeight.notna()]
    y1, y2 = sorted(df.year.unique())
    pan = df.groupby(['PitcherId','year']).apply(features, include_groups=False).reset_index()
    pan = pan[pan.n >= MIN_P]
    a = pan[pan.year == y1].set_index('PitcherId')
    b = pan[pan.year == y2].set_index('PitcherId')
    both = a.join(b, lsuffix='_1', rsuffix='_2', how='inner')
    print(f"\n===== {label}  (pitchers in both yrs, >={MIN_P} FF: n={len(both)}) =====")
    se = 1/np.sqrt(len(both)-3)
    print(f"approx SE of r ~ {se:.3f}")
    print(f"{'feature':<12} {'reliability':>11} {'validity':>9} {'partial|waste':>13}")
    crit = both.adjT_2
    w1 = both.waste_pct_1
    for f in FEATS:
        rel = both[f+'_1'].corr(both[f+'_2'])
        val = both[f+'_1'].corr(crit)
        # partial validity controlling waste_pct (skip for waste itself/subtypes)
        if f.startswith('waste'):
            part = ''
        else:
            r_xy, r_xw, r_yw = val, both[f+'_1'].corr(w1), crit.corr(w1)
            part = f"{(r_xy - r_xw*r_yw)/np.sqrt((1-r_xw**2)*(1-r_yw**2)):+.3f}"
        print(f"{f:<12} {rel:>+11.3f} {val:>+9.3f} {part:>13}")
    # criterion self-reliability for context
    print(f"{'adjT (crit)':<12} {both.adjT_1.corr(both.adjT_2):>+11.3f}")
    # per-pitch cost of each waste type (year-1 pitches, pooled)
    d1 = df[df.year == y1]
    x, z = d1.PlateLocSide.values, d1.PlateLocHeight.values
    ax = np.abs(x)
    heart = (ax <= 0.558) & (z >= 1.83) & (z <= 3.17)
    shadow = (ax <= 1.108) & (z >= 1.17) & (z <= 3.83) & ~heart
    chase = (ax <= 1.658) & (z >= 0.5) & (z <= 4.5) & ~heart & ~shadow
    waste = ~(heart | shadow | chase)
    low = waste & (z < 0.5); high = waste & (z > 4.5); horiz = waste & ~low & ~high
    base = d1.adjT.mean()
    print(f"\nper-pitch mean adjT ({y1}, all D1 FF; overall {base:+.4f}; lower=better):")
    for name, m in [('heart',heart),('shadow',shadow),('chase',chase),
                    ('waste-low',low),('waste-high',high),('waste-horiz',horiz)]:
        v = d1.adjT.values[m]
        print(f"  {name:<11} n={m.sum():>7}  share={m.mean():>6.1%}  adjT={v.mean():+.4f}  vs base {v.mean()-base:+.4f}")
