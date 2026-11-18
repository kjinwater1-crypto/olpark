"""Train LightGBM for hourly main-lot occupancy; evaluate on 2025-07..12 holdout vs baselines."""
import pandas as pd, numpy as np, lightgbm as lgb, json

D = './data/'
CAP = 3600          # reference capacity for main lots (observed max 3,578 on 2024-09-29)
V = ['KSPO', 'HANDBALL', 'OLHALL', 'ARTHALL', 'MUSE', 'OUTDOOR']
FEATS = (['hour', 'dow', 'is_hol', 'is_off', 'next_off', 'prev_off', 'month', 'doy_sin', 'doy_cos', 'school_vac',
          'avg_temp', 'max_temp', 'min_temp', 'rain_mm', 'rainy', 'heavy_rain', 'avg_cloud',
          'laud_total', 'press_total', 'n_venues', 'aud_big3']
         + [f'laud_{v}' for v in V] + [f'rel_{v}' for v in V] + [f'press_{v}' for v in V])

f = pd.read_pickle(D + 'hourly.pkl')
f = f[f.valid == 1].copy()
f['daytype'] = np.where(f.is_hol == 1, 6, f.dow)            # holidays behave like Sundays
f['event_day'] = (f.aud_total >= 1000).astype(int)
f['big_event_day'] = (f.aud_big3 >= 5000).astype(int)
split = pd.Timestamp('2025-07-01')
tr, te = f[f.ts < split].copy(), f[f.ts >= split].copy()


def level(x):
    r = np.asarray(x) / CAP
    return np.select([r >= 0.9, r >= 0.7, r >= 0.5], [3, 2, 1], 0)


def evaluate(y, p, frame, name):
    e = np.abs(y - p)
    day = frame.assign(y=y, p=p).groupby('date').agg(y=('y', 'max'), p=('p', 'max'), ev=('big_event_day', 'max'))
    m = {
        'model': name,
        'MAE': e.mean(),
        'MAE_daytime(9-22h)': e[frame.hour.between(9, 22)].mean(),
        'MAE_big_event_days': e[frame.big_event_day == 1].mean(),
        'MAE_other_days': e[frame.big_event_day == 0].mean(),
        'MAPE(occ>=500)': (e[y >= 500] / y[y >= 500]).mean() * 100,
        'R2': 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        'daily_peak_MAE': (day.y - day.p).abs().mean(),
        'daily_peak_MAPE': ((day.y - day.p).abs() / day.y).mean() * 100,
        'daily_peak_MAE_event': (day.y - day.p)[day.ev == 1].abs().mean(),
        'level_acc(hourly,9-22h)': (level(y) == level(p))[frame.hour.between(9, 22).values].mean() * 100,
    }
    # congestion alarm (>=70% of reference capacity) at hour level
    a, b = y >= 0.7 * CAP, p >= 0.7 * CAP
    m['alarm_recall'] = (a & b).sum() / max(a.sum(), 1) * 100
    m['alarm_precision'] = (a & b).sum() / max(b.sum(), 1) * 100
    m['n_alarm_hours'] = int(a.sum())
    return m


results = []
y_te = te.occ_main.values
# Baseline 1: day-type x hour mean
b1 = tr.groupby(['daytype', 'hour']).occ_main.mean()
p1 = te.set_index(['daytype', 'hour']).index.map(b1).values
results.append(evaluate(y_te, p1, te, 'B1 요일x시간 평균'))
# Baseline 2: month x day-type x hour mean (last year same month)
b2 = tr.groupby(['month', 'daytype', 'hour']).occ_main.mean()
p2 = np.array([b2.get(k, np.nan) for k in zip(te.month, te.daytype, te.hour)])
p2 = np.where(np.isnan(p2), p1, p2)
results.append(evaluate(y_te, p2, te, 'B2 월x요일x시간 평균'))

params = dict(objective='regression', learning_rate=0.03, num_leaves=63, min_data_in_leaf=30,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
# early-stopping inside the training window (last 3 months of train as validation)
inner = pd.Timestamp('2025-04-01')
dtr = lgb.Dataset(tr.loc[tr.ts < inner, FEATS], tr.loc[tr.ts < inner, 'occ_main'])
dva = lgb.Dataset(tr.loc[tr.ts >= inner, FEATS], tr.loc[tr.ts >= inner, 'occ_main'])
m0 = lgb.train(params, dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
best_it = max(m0.best_iteration, 300)
model = lgb.train(params, lgb.Dataset(tr[FEATS], tr.occ_main), best_it)
p3 = np.clip(model.predict(te[FEATS]), 0, None)
results.append(evaluate(y_te, p3, te, 'LightGBM(제안)'))

# ablation: no event features
FE_NOEV = [c for c in FEATS if not any(c.startswith(s) for s in ('laud', 'press', 'rel_', 'n_venues', 'aud_'))]
m_ne = lgb.train(params, lgb.Dataset(tr[FE_NOEV], tr.occ_main), best_it)
p4 = np.clip(m_ne.predict(te[FE_NOEV]), 0, None)
results.append(evaluate(y_te, p4, te, 'LightGBM(행사 변수 제외)'))

res = pd.DataFrame(results).set_index('model')
pd.set_option('display.width', 250)
print(res.round(1).T.to_string())
print('best_iteration', best_it)
imp = pd.Series(model.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
print((imp / imp.sum() * 100).round(1).head(20).to_string())
te['pred'] = p3; te['b1'] = p1; te['b2'] = p2; te['pred_noev'] = p4
te[['ts', 'date', 'hour', 'dow', 'is_hol', 'occ_main', 'pred', 'b1', 'b2', 'pred_noev', 'aud_total', 'aud_big3', 'big_event_day']].to_pickle(D + 'holdout_pred.pkl')
res.to_csv(D + 'holdout_metrics.csv')
json.dump({'best_it': int(best_it), 'feats': FEATS, 'cap': CAP}, open(D + 'model_meta.json', 'w'))
model.save_model(D + 'lgb_holdout.txt')
