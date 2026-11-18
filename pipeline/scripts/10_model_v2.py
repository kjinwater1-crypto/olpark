"""Model v2: recency-level features (7-day-ahead operational mode) + P80 quantile model for alarms."""
import pandas as pd, numpy as np, lightgbm as lgb, json
exec(open('./scripts/09_model.py', encoding='utf-8').read().split("results = []")[0])  # reuse data prep + helpers

def add_recency(fr, hist):
    """mean occupancy for same hour & day-group over [d-35, d-7]; mean daily peak over same window."""
    hist = hist.copy()
    hist['grp'] = np.select([hist.is_hol == 1, hist.dow == 6, hist.dow == 5], [2, 2, 1], 0)
    daily = hist.groupby(['date', 'grp']).occ_main.max().reset_index(name='dpk')
    hourly = hist[['date', 'grp', 'hour', 'occ_main']]
    out = fr.copy()
    out['grp'] = np.select([out.is_hol == 1, out.dow == 6, out.dow == 5], [2, 2, 1], 0)
    dates = sorted(out.date.unique())
    rec_h, rec_d = {}, {}
    hd = hourly.set_index('date').sort_index()
    dd = daily.set_index('date').sort_index()
    for d in dates:
        lo, hi = d - pd.Timedelta(days=35), d - pd.Timedelta(days=7)
        wh = hd.loc[lo:hi]
        wd = dd.loc[lo:hi]
        if len(wd) < 10:
            continue
        for g, s in wh.groupby('grp'):
            for h, v in s.groupby('hour').occ_main.mean().items():
                rec_h[(d, g, h)] = v
        for g, v in wd.groupby('grp').dpk.mean().items():
            rec_d[(d, g)] = v
    out['rec_hour'] = [rec_h.get((d, g, h), np.nan) for d, g, h in zip(out.date, out.grp, out.hour)]
    out['rec_dpk'] = [rec_d.get((d, g), np.nan) for d, g in zip(out.date, out.grp)]
    return out

params = dict(objective='regression', learning_rate=0.03, num_leaves=63, min_data_in_leaf=30,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
allf = pd.concat([tr, te])
allf = add_recency(allf, allf)
tr2, te2 = allf[allf.ts < split].copy(), allf[allf.ts >= split].copy()
F2 = FEATS + ['rec_hour', 'rec_dpk']

res = []
y = te2.occ_main.values
mA = lgb.train(params, lgb.Dataset(tr2[FEATS], tr2.occ_main), 763)
pA = np.clip(mA.predict(te2[FEATS]), 0, None)
res.append(evaluate(y, pA, te2, 'A: 장기예측(행사+달력+날씨)'))
mB = lgb.train(params, lgb.Dataset(tr2[F2], tr2.occ_main), 763)
pB = np.clip(mB.predict(te2[F2]), 0, None)
res.append(evaluate(y, pB, te2, 'B: 운영(최근 4주 수준 추가, 7일 전 예측)'))

# P80 quantile models for alarms
qp = dict(params, objective='quantile', alpha=0.8)
qA = lgb.train(qp, lgb.Dataset(tr2[FEATS], tr2.occ_main), 763)
qB = lgb.train(qp, lgb.Dataset(tr2[F2], tr2.occ_main), 763)
pqA = np.clip(qA.predict(te2[FEATS]), 0, None)
pqB = np.clip(qB.predict(te2[F2]), 0, None)
for nm, pq, pm in [('A', pqA, pA), ('B', pqB, pB)]:
    a = y >= 0.7 * CAP
    b = pq >= 0.7 * CAP
    print(f'{nm} P80 alarm: recall {(a & b).sum() / a.sum() * 100:.1f}%  precision {(a & b).sum() / max(b.sum(), 1) * 100:.1f}%  alarms {b.sum()}  coverage(y<=P80) {(y <= pq).mean() * 100:.1f}%')
    # day-level alarm: did the day reach >=70%? predicted day max P80 >= 70%?
    dd = te2.assign(y=y, q=pq).groupby('date').agg(y=('y', 'max'), q=('q', 'max'))
    A_, B_ = dd.y >= 0.7 * CAP, dd.q >= 0.7 * CAP
    print(f'   day-level alarm: recall {(A_ & B_).sum() / A_.sum() * 100:.1f}% ({(A_ & B_).sum()}/{A_.sum()} days), precision {(A_ & B_).sum() / max(B_.sum(), 1) * 100:.1f}% ({B_.sum()} alarm days)')
    A9, B9 = dd.y >= 0.85 * CAP, dd.q >= 0.85 * CAP
    print(f'   day-level >=85%: recall {(A9 & B9).sum()}/{A9.sum()}, alarm days {B9.sum()}')

r = pd.DataFrame(res).set_index('model')
pd.set_option('display.width', 250)
print(r.round(1).T.to_string())
te2['pA'], te2['pB'], te2['qA'], te2['qB'] = pA, pB, pqA, pqB
te2.to_pickle(D + 'holdout_pred_v2.pkl')
r.to_csv(D + 'holdout_metrics_v2.csv')
# monthly bias
print(te2.assign(eA=pA - y, eB=pB - y).groupby(te2.ts.dt.month)[['eA', 'eB']].mean().round(0).to_string())
