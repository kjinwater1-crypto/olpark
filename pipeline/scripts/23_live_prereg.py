"""Pre-registered check of the forecasts published on GitHub, scored by the rule in live/SCORING.md (published 2026-11-18 22:00).

- v1: published 2026-11-15 06:23 (commit e80540a, growth 26%)
- v2: published 2026-11-18 20:55 (commit 2d2f429, growth 14%)
- v3: published 2026-11-18 21:59 (commit d815bde, growth 3%), the main result from 11-19
- simple baseline: mean of 2025-07..12 for the same day type (weekday, Saturday, Sunday or holiday) and hour
- live: the park homepage's real-time count (north 2, east 2, south 2 gates), every 20 minutes by GitHub Actions
  (branch 'live', data/live_rt_log_github.csv), plus the browser log of 11-18 evening.
The live count keeps long-stay cars without an exit record, so each day is aligned on its 03 h:
adjusted = raw - (that day's 03 h reading) + (the forecast's 03 h value). Hourly values are the max of the readings in the hour,
like the forecast's hourly peak. Scored: 2026-11-19..11-30, 10-22 h, only days whose scoring window has passed
(a reading at 22 h or later, or a reading from a later day). Writes data/live_prereg.json and data/live_prereg_summary.json;
the summary carries the overall metrics, the metrics by day and by group (days the main version alerted, other days),
and two sentences for the proposal and the web page (numbers rounded half up, as the web page shows them)."""
import json, math, os, pandas as pd, numpy as np
D = './data/'; W = './data/published/'
T = 2520
VERS = {'v1': W + 'app_data_published_20261115.json', 'v2': W + 'app_data_published_20261118.json', 'v3': W + 'app_data_published_20261118b.json'}
MAIN, START, END, H0, H1 = 'v3', '2026-11-19', '2026-11-30', 10, 22
# supplementary rule added to live/SCORING.md on 2026-11-23: noon update of the main version from that day's 10 h and 11 h
NOON_FROM = os.environ.get('OLPARK_NOON_FROM', '2026-11-24')      # the env value is only for a dry run on earlier days
NOON_HRS, NOON_W, NOON_CLIP, NOON_H0 = (10, 11), 0.5, (0.6, 1.4), 12
r0 = lambda v: int(math.floor(v + 0.5))
fmt = lambda v: f'{r0(v):,}'

logs = [pd.read_csv(D + f) for f in ('live_rt_log_github.csv', 'live_rt_log_browser.csv') if os.path.exists(D + f)]
r = pd.concat(logs, ignore_index=True)
r['ts'] = pd.to_datetime(r.ts_kst); r['date'] = r.ts.dt.strftime('%Y-%m-%d'); r['h'] = r.ts.dt.hour
last_day = r.date.max()
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

def metrics(L):
    x = pd.DataFrame(L)
    m = dict(n=len(x), mae=round(float((x.adj - x.p).abs().mean()), 1), bias=round(float((x.p - x.adj).mean()), 1))
    if x.lo.notna().any():
        m['coverage'] = round(float(((x.adj >= x.lo) & (x.adj <= x.hi)).mean() * 100), 1)
    return m

rows, per, per_noon, days = [], [], [], []
for d, g in hr.groupby('date'):
    if not (START <= d <= END):
        continue
    if not (g.h.max() >= H1 or d < last_day):          # the day's scoring window has not passed yet
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
            per.append(dict(k=k, d=d, h=x.h, adj=adj, p=p[x.h], lo=lo[x.h] if lo else None, hi=hi[x.h] if hi else None))
    if MAIN in fc:
        p, lo, hi = fc[MAIN]
        for x in g.itertuples():
            rows.append(dict(d=d, t=f'{x.h:02d}:00', raw=int(x.raw), api=int(round(x.raw - raw3 + p[3])), p=p[x.h], lo=lo[x.h], hi=hi[x.h], v=MAIN,
                             s=int(H0 <= x.h <= H1), b=r0(fc['base'][0][x.h])))
        act = [x.raw - raw3 + p[3] for x in sc.itertuples()]
        dd = dict(d=d, act_max=r0(max(act)) if act else None,
                  ev=[dict(vn=e.get('vn'), t=e.get('t'), s=e.get('s')) for e in pubs[MAIN][d].get('ev', [])])
        for k in pubs:
            if d in pubs[k]:
                pk, hk = max(pubs[k][d]['p']), max(pubs[k][d]['hi'])
                dd[k] = dict(p_max=pk, alert='경보' if pk >= T else ('주의' if hk >= T else ''))
        am = {x.h: x.raw - raw3 + p[3] for x in g.itertuples()}
        if d >= NOON_FROM and all(hh in am for hh in NOON_HRS):
            ratio = sum(am[hh] for hh in NOON_HRS) / sum(p[hh] for hh in NOON_HRS)
            f = min(max(1 + NOON_W * (ratio - 1), NOON_CLIP[0]), NOON_CLIP[1])
            pm = [x for x in sc.itertuples() if x.h >= NOON_H0]
            for x in pm:
                a = x.raw - raw3 + p[3]
                per_noon.append(dict(k='noon', d=d, h=x.h, adj=a, p=p[x.h] * f, lo=lo[x.h] * f, hi=hi[x.h] * f))
                per_noon.append(dict(k=MAIN, d=d, h=x.h, adj=a, p=p[x.h], lo=lo[x.h], hi=hi[x.h]))
            hrs = range(NOON_H0, H1 + 1)
            npk, nhk = max(p[hh] for hh in hrs) * f, max(hi[hh] for hh in hrs) * f
            dd['noon'] = dict(ratio=round(ratio, 3), f=round(f, 3), p_max=r0(npk), alert='경보' if npk >= T else ('주의' if nhk >= T else ''),
                              act_max_pm=r0(max(x.raw - raw3 + p[3] for x in pm)) if pm else None)
            for rw in rows:
                if rw['d'] == d and int(rw['t'][:2]) >= NOON_H0:
                    hh = int(rw['t'][:2]); rw.update(pn=r0(p[hh] * f), lon=r0(lo[hh] * f), hin=r0(hi[hh] * f))
        days.append(dd)

P = pd.DataFrame(per)
summ = dict(rule='live/SCORING.md', main=MAIN, period=[START, END], hours=[H0, H1], n_days=len(days), days=days, metrics={}, by_group={})
if len(P):
    for k, g in P.groupby('k', sort=False):
        summ['metrics'][k] = metrics(g.to_dict('records'))
    for e in days:
        e['m'] = {k: metrics(g.to_dict('records')) for k, g in P[P.d == e['d']].groupby('k', sort=False)}
    alert_days = {e['d'] for e in days if e.get(MAIN, {}).get('alert')}
    for name, sel in (('alert', P.d.isin(alert_days)), ('calm', ~P.d.isin(alert_days))):
        if sel.any():
            summ['by_group'][name] = dict(n_days=int(P[sel].d.nunique()), **{k: metrics(g.to_dict('records')) for k, g in P[sel].groupby('k', sort=False)})
if per_noon:
    Q = pd.DataFrame(per_noon)
    nd = [e for e in days if 'noon' in e]
    summ['noon'] = dict(rule='live/SCORING.md, 2026-11-23 보충 채점', since=NOON_FROM, hours=[NOON_H0, H1], n_days=int(Q.d.nunique()),
                        metrics={k: metrics(g.to_dict('records')) for k, g in Q.groupby('k', sort=False)},
                        hits=sum(1 for e in nd if e['noon']['alert'] and e['noon']['act_max_pm'] >= T),
                        alerts=sum(1 for e in nd if e['noon']['alert']),
                        actual=sum(1 for e in nd if e['noon']['act_max_pm'] is not None and e['noon']['act_max_pm'] >= T))

def md(s, with_month=True):
    m, dd = int(s[5:7]), int(s[8:10])
    return f'{m}월 {dd}일' if with_month else f'{dd}일'

def and_days(ds):
    """11월 21일과 22일 / 11월 21일, 22일과 28일 (month written once while it does not change)"""
    out = [md(d, i == 0 or ds[i - 1][5:7] != d[5:7]) for i, d in enumerate(ds)]
    return out[0] if len(out) == 1 else ', '.join(out[:-1]) + '과 ' + out[-1]

def ndays(n, topic=True):
    """하루는, 이틀은, 사흘은, 나흘은, 5일은 (topic=False: no particle)"""
    w = {1: '하루', 2: '이틀', 3: '사흘', 4: '나흘'}.get(n, f'{n}일')
    return w + (('는' if n == 1 else '은') if topic else '')

def and_nums(vs):
    out = [f'{fmt(v)}대' for v in vs]
    return out[0] if len(out) == 1 else ', '.join(out[:-1]) + '와 ' + out[-1]

if MAIN in summ['metrics'] and days:
    mm, bb = summ['metrics'][MAIN], summ['metrics'].get('base')
    d0, d1 = days[0]['d'], days[-1]['d']
    span = f'{int(d0[5:7])}월 {int(d0[8:10])}~{int(d1[8:10])}일' if d0[5:7] == d1[5:7] else f'{md(d0)}~{md(d1)}'
    if bb:
        diff = (mm['mae'] - bb['mae']) / bb['mae'] * 100
        cmp = '와 비슷했고' if abs(diff) < 5 else (f'보다 {r0(-diff)}% 작았고' if diff < 0 else f'보다 {r0(diff)}% 컸고')
        cmp_short = '단순 기준과 비슷했지만' if abs(diff) < 5 else (f'단순 기준보다 {r0(-diff)}% 작았지만' if diff < 0 else f'단순 기준보다 {r0(diff)}% 컸고')
        s1 = f"{span}을 채점하니 셋째 판의 평균 오차는 {fmt(mm['mae'])}대로 단순 기준 {fmt(bb['mae'])}대{cmp}, 실측의 {r0(mm['coverage'])}%가 80% 범위 안에 들었습니다."
    else:
        s1 = f"{span}을 채점하니 셋째 판의 평균 오차는 {fmt(mm['mae'])}대이고, 실측의 {r0(mm['coverage'])}%가 80% 범위 안에 들었습니다."
    al = [e for e in days if e.get(MAIN, {}).get('alert')]
    hit = [e for e in al if e['act_max'] is not None and e['act_max'] >= T]
    miss = [e for e in days if not e.get(MAIN, {}).get('alert') and e['act_max'] is not None and e['act_max'] >= T]
    kinds = sorted({e[MAIN]['alert'] for e in al}, key=['주의', '경보'].index)
    kind = '와 '.join(kinds) if len(kinds) == 2 else (kinds[0] if kinds else '')
    calm = summ['by_group'].get('calm', {}).get(MAIN)
    s2 = ''
    if al:
        cb = summ['by_group'].get('calm', {}).get('base')
        lead = ''
        if calm and cb:
            better = calm['mae'] < cb['mae'] * 0.95
            cmpc = '보다 작았' if better else ('와 비슷했' if calm['mae'] <= cb['mae'] * 1.05 else '보다 컸')
            lead = (f"알림이 없던 {ndays(summ['by_group']['calm']['n_days'])} 평균 오차가 {fmt(calm['mae'])}대로 단순 기준 {fmt(cb['mae'])}대{cmpc}"
                    + ('지만, ' if better and not hit else '고, '))
        if not hit and len(al) <= 3:
            over = [r0(e[MAIN]['p_max'] - e['act_max']) for e in al]
            s2 = (f"{lead}{kind}를 낸 {and_days([e['d'] for e in al])}은 실측 하루 최대가 {and_nums([e['act_max'] for e in al])}로 혼잡 기준 {T:,}대에 못 미쳐 "
                  f"예보 정점이 {and_nums(over)} 높았습니다.")
        elif len(hit) == len(al):
            s2 = f"{lead}경보나 주의를 낸 {len(al)}일은 모두 실측 하루 최대가 혼잡 기준 {T:,}대를 넘었습니다."
        else:
            s2 = f"{lead}경보나 주의를 낸 {len(al)}일 가운데 {len(hit)}일은 실측 하루 최대가 혼잡 기준 {T:,}대를 넘었고 {len(al) - len(hit)}일은 못 미쳤습니다."
    s3 = '알림 없이 혼잡 기준을 넘은 날은 없었습니다.' if not miss else f'알림 없이 혼잡 기준을 넘은 날이 {len(miss)}일 있었습니다.'
    summ['sentence'] = ' '.join(x for x in (s1, s2, s3) if x)
    # short form for the one-page summary
    if al and not hit and bb:
        summ['sentence_short'] = (f"{span} 채점에서 셋째 판의 평균 오차는 {fmt(mm['mae'])}대로 {cmp_short}, "
                                  f"{kind}를 낸 공연일 {ndays(len(al))} 실측이 혼잡 기준에 못 미쳐 정점을 높게 봤습니다.")
    else:
        summ['sentence_short'] = (f"{span} 채점에서 셋째 판의 평균 오차는 {fmt(mm['mae'])}대이고, 경보나 주의를 낸 {len(al)}일 가운데 {len(hit)}일이 실제로 혼잡 기준을 넘었습니다.")
    summ['scored_through'] = d1
json.dump(rows, open(D + 'live_prereg.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
json.dump(summ, open(D + 'live_prereg_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != 'days'}, ensure_ascii=False, indent=1)[:3000])
