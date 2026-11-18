"""Pre-registered check of the forecasts published on GitHub, scored by the rule in live/SCORING.md (published 2026-11-18 22:00).

- v1: published 2026-11-15 06:23 (commit e80540a, growth 26%)
- v2: published 2026-11-18 20:55 (commit 2d2f429, growth 14%)
- v3: published 2026-11-18 21:59 (commit d815bde, growth 3%), the main result from 11-19
- simple baseline: mean of 2025-07..12 for the same day type (weekday, Saturday, Sunday or holiday) and hour
- live: the park homepage's real-time count (north 2, east 2, south 2 gates), every 20 minutes by GitHub Actions
  (branch 'live', data/live_rt_log_github.csv), plus the browser log of 11-18 evening.
The live count keeps long-stay cars without an exit record, so each day is aligned on its 03 h:
adjusted = raw - (that day's 03 h reading) + (the forecast's 03 h value). Hourly values are the max of the readings in the hour,
like the forecast's hourly peak. Scored: 2026-11-19..11-30, 10-22 h. Writes data/live_prereg.json and data/live_prereg_summary.json."""
import json, os, pandas as pd, numpy as np
D = './data/'; W = './data/published/'
T = 2520
VERS = {'v1': W + 'app_data_published_20261115.json', 'v2': W + 'app_data_published_20261118.json', 'v3': W + 'app_data_published_20261118b.json'}
MAIN, START, END, H0, H1 = 'v3', '2026-11-19', '2026-11-30', 10, 22

logs = [pd.read_csv(D + f) for f in ('live_rt_log_github.csv', 'live_rt_log_browser.csv') if os.path.exists(D + f)]
r = pd.concat(logs, ignore_index=True)
r['ts'] = pd.to_datetime(r.ts_kst); r['date'] = r.ts.dt.strftime('%Y-%m-%d'); r['h'] = r.ts.dt.hour
hr = r.groupby(['date', 'h']).parking_now.max().rename('raw').reset_index()
pubs = {k: {x['d']: x for x in json.load(open(f, encoding='utf-8'))['days']} for k, f in VERS.items() if os.path.exists(f)}

# simple baseline: 2025 H2 mean by day type and hour
h = pd.read_pickle(D + 'hourly.pkl'); h['date'] = pd.to_datetime(h['date'])
h = h[(h.valid == 1) & (h.date >= '2025-07-01') & (h.date <= '2025-12-30')]
h['dtype'] = np.where(h.is_hol == 1, 6, np.where(h.dow == 5, 5, np.where(h.dow == 6, 6, 0)))
base = h.groupby(['dtype', 'hour']).occ_main.mean()
def base_day(d):
    t = pd.Timestamp(d); k = 5 if t.dayofweek == 5 else 6 if t.dayofweek == 6 else 0
    return [float(base[(k, hh)]) for hh in range(24)]

rows, per = [], {k: [] for k in list(pubs) + ['base']}
days = []
for d, g in hr.groupby('date'):
    if not (START <= d <= END):
        continue
    b3 = g[g.h == 3]
    if b3.empty:
        continue
    raw3 = int(b3.raw.iloc[0])
    sc = g[g.h.between(H0, H1)]
    fc = {k: (pubs[k][d]['p'], pubs[k][d]['lo'], pubs[k][d]['hi']) for k in pubs if d in pubs[k]}
    fc['base'] = (base_day(d), None, None)
    for k, (p, lo, hi) in fc.items():
        for x in sc.itertuples():
            adj = x.raw - raw3 + p[3]
            per[k].append(dict(d=d, h=x.h, adj=adj, p=p[x.h], lo=lo[x.h] if lo else None, hi=hi[x.h] if hi else None))
    if MAIN in fc:
        p, lo, hi = fc[MAIN]
        for x in g.itertuples():
            rows.append(dict(d=d, t=f'{x.h:02d}:00', raw=int(x.raw), api=int(round(x.raw - raw3 + p[3])), p=p[x.h], lo=lo[x.h], hi=hi[x.h], v=MAIN,
                             s=int(H0 <= x.h <= H1)))
        dd = dict(d=d, act_max=int(round(max(x.raw for x in sc.itertuples()) - raw3 + p[3])) if len(sc) else None)
        for k in pubs:
            if d in pubs[k]:
                pk, hk = max(pubs[k][d]['p']), max(pubs[k][d]['hi'])
                dd[k] = dict(p_max=pk, alert='경보' if pk >= T else ('주의' if hk >= T else ''))
        days.append(dd)

summ = dict(rule='live/SCORING.md', main=MAIN, period=[START, END], hours=[H0, H1], n_days=len(days), days=days, metrics={})
for k, L in per.items():
    if not L:
        continue
    x = pd.DataFrame(L)
    m = dict(n=len(x), mae=round(float((x.adj - x.p).abs().mean()), 1), bias=round(float((x.p - x.adj).mean()), 1))
    if x.lo.notna().any():
        m['coverage'] = round(float(((x.adj >= x.lo) & (x.adj <= x.hi)).mean() * 100), 1)
    summ['metrics'][k] = m
if MAIN in summ['metrics']:
    mm = summ['metrics'][MAIN]; big = [e for e in days if e.get(MAIN, {}).get('alert')]
    hit = sum(1 for e in big if e['act_max'] is not None and e['act_max'] >= T)
    summ['sentence'] = (f"11월 19일부터 {days[-1]['d'][5:7].lstrip('0')}월 {days[-1]['d'][8:].lstrip('0')}일까지 {len(days)}일의 10~22시를 미리 공개한 규칙대로 채점한 결과, "
                        f"공연 전에 공개한 예보의 평균 오차는 {mm['mae']:.0f}대이고 실측의 {mm.get('coverage', 0):.0f}%가 80% 예측 범위 안에 들었습니다. "
                        f"경보나 주의를 낸 {len(big)}일 중 {hit}일은 실제로 혼잡 기준을 넘었습니다.")
json.dump(rows, open(D + 'live_prereg.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
json.dump(summ, open(D + 'live_prereg_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != 'days'}, ensure_ascii=False, indent=1)[:2500])
