"""Assemble compact JSON for the web app: forecasts, what-if response curves, exit queue params, holdout, history."""
import pandas as pd, numpy as np, json, lightgbm as lgb, duckdb, os

D = './data/'
CAP = 3600
TODAY = os.environ.get('OLPARK_TODAY', '2026-11-14')        # date the forecast is published (20_daily_forecast.py sets it)
meta = json.load(open(D + 'final_meta.json'))
FF, OFF = meta['FF'], meta['OFF']
q10 = {int(k): v for k, v in meta['q10'].items()}; q90 = {int(k): v for k, v in meta['q90'].items()}
models = [lgb.Booster(model_file=D + f'final_model_{i}.txt') for i in range(3)]
V = ['KSPO', 'HANDBALL', 'OLHALL', 'ARTHALL', 'MUSE', 'OUTDOOR']
VN = {'KSPO': 'KSPO DOME', 'HANDBALL': '핸드볼경기장', 'OLHALL': '올림픽홀', 'ARTHALL': '우리금융아트홀', 'MUSE': '뮤즈라이브', 'OUTDOOR': '야외(88잔디마당 등)'}

def predict(frame):
    r = np.mean([m.predict(frame[FF]) for m in models], axis=0)
    return np.clip(r * (frame.rec_hour.values + OFF), 0, None)

def band(p):
    b = pd.cut(pd.Series(p), [-1, 300, 800, 1500, 2200, 1e9], labels=False).values
    return np.clip(np.array([(x + OFF) * q10[k] - OFF for x, k in zip(p, b)]), 0, None), np.array([(x + OFF) * q90[k] - OFF for x, k in zip(p, b)])

fut = pd.read_pickle(D + 'future_forecast.pkl')
ed = pd.read_csv(D + 'event_days_long_settled.csv', parse_dates=['date'])
ev_meta = pd.read_csv(D + 'events_parsed.csv', dtype={'plcCd': str})
sd = pd.read_csv(D + 'show_days.csv', parse_dates=['date'])
sd = sd.merge(ev_meta[['evtSeq', 'title', 'plc', 'plcCd', 'ndTm', 'method']], on='evtSeq', how='left')
import importlib.util
spec = importlib.util.spec_from_file_location('ef', './scripts/07_event_features.py')
src = open('./scripts/07_event_features.py', encoding='utf-8').read()
ns = {}
exec(src.split("ev['venue'] =")[0], ns)
sd['venue'] = [ns['venue_from_sched'](p, c) for p, c in zip(sd.plc, sd.plcCd)]

def hhmm(m):
    m = int(m); return f'{m // 60:02d}:{m % 60:02d}'

days = []
for d, g in fut.groupby('date'):
    evs = sd[sd.date == d].sort_values('first_start')
    g = g.sort_values('hour')
    g0 = g.iloc[0]
    el = [dict(v=r.venue, vn=VN.get(r.venue, r.plc), t=str(r.title)[:60], s=hhmm(r.first_start), n=int(r.n_shows),
               a=int(round(float(g0.get(f'aud_{r.venue}', 0) or 0) / max(1, len(evs[evs.venue == r.venue])))),
               **({'u': 1} if r.method == 'default' else {}), **({'m': 1} if str(r.evtSeq).startswith('M') else {})) for r in evs.itertuples()]   # m: announced show not yet in the public list   # u: start time not published, typical time assumed
    days.append(dict(d=d.strftime('%Y-%m-%d'), w=int(g.dow.iloc[0]), h=int(g.is_hol.iloc[0]), ev=el,
                     p=[int(round(x)) for x in g.p], lo=[int(round(x)) for x in g.lo], hi=[int(round(min(x, CAP))) for x in g.hi]))

# ---- days before the publish date keep the forecast that was public at the time (pre-registered values)
PUB = os.environ.get('OLPARK_PUBLISHED')
if PUB and os.path.exists(PUB):
    pub = {x['d']: x for x in json.load(open(PUB, encoding='utf-8'))['days']}
    days = [pub[x['d']] if x['d'] < TODAY and x['d'] in pub else x for x in days]

# ---- default audiences per venue (settled-attendance medians, descriptive)
med = ed[ed.src == 'rent+sched'].groupby('venue').aud.median().to_dict()
resp_meta = {v: int(round(med.get(v, 0))) for v in ['KSPO', 'HANDBALL', 'OLHALL', 'OUTDOOR']}

# ---- the date after which the public schedule is sparse: big-venue shows continue with gaps of at most 14 days until then
big_dates = sorted({pd.Timestamp(x['d']) for x in days if any(e['v'] in ('KSPO', 'HANDBALL', 'OLHALL') for e in x['ev']) and x['d'] >= TODAY})
sched_horizon = big_dates[0] if big_dates else pd.Timestamp(TODAY)
for a_, b_ in zip(big_dates, big_dates[1:]):
    if (b_ - a_).days > 14:
        break
    sched_horizon = b_
sched_horizon = sched_horizon.strftime('%Y-%m-%d')

# ---- exit queue parameters
ins = json.load(open(D + 'insights.json'))
XC = json.load(open(D + 'exit_clear.json'))          # 25_exit_clear.py
ex_in = np.clip(np.array(ins['arrival']['excess_in_profile'][:37]), 0, None).sum()   # -360..0 min
w = pd.read_csv(D + 'event_days_wide_settled.csv', parse_dates=['date'])
hourly = pd.read_pickle(D + 'hourly_settled.pkl')
offd = hourly[(hourly.is_off == 1) & (hourly.valid == 1)][['date']].drop_duplicates()
kd = w.merge(offd, on='date')
kd = kd[(kd.aud_KSPO >= 8000) & kd.start_KSPO.between(960, 1140)]
cars_per_1000 = ex_in / kd.aud_total.mean() * 1000
# observed: largest 30-minute exit volume after big KSPO shows (main gates), median over show days
# (needs the 10-minute table of 06_occ_main.py; the value is cached in queue_obs.json so weekly runs work without the database)
if os.path.exists(D + 'olpark.duckdb'):
    con = duckdb.connect(D + 'olpark.duckdb', read_only=True)
    sm = con.execute("SELECT slot, n_out FROM slot_main").df(); sm['date'] = sm.slot.dt.normalize(); sm['mins'] = sm.slot.dt.hour * 60 + sm.slot.dt.minute
    kk = w[(w.aud_KSPO >= 8000) & w.start_KSPO.between(960, 1200)][['date', 'start_KSPO']]
    ex = sm.merge(kk, on='date'); ex['rel'] = ex.mins - ex.start_KSPO
    obs30 = float(ex[ex.rel.between(60, 400)].groupby('date').n_out.apply(lambda x: x.rolling(3).sum().max()).median())
    # the cars behind those exits: KSPO DOME plus Olympic Hall and Ticketlink Live Arena shows starting within 60 minutes of it
    cd = w[w.date.isin(ex.date.unique()) & (w.aud_KSPO >= 8000) & w.start_KSPO.between(960, 1200)].copy()
    cd['wave'] = cd.aud_KSPO + sum(np.where((cd[f'start_{v}'] - cd.start_KSPO).abs() <= 60, cd[f'aud_{v}'].fillna(0), 0) for v in ('OLHALL', 'HANDBALL'))
    wave_med = float(cd.wave.median())
    json.dump({'obs_max30': obs30, 'wave_median': wave_med, 'n_days': int(len(cd))}, open(D + 'queue_obs.json', 'w'))
else:
    _q = json.load(open(D + 'queue_obs.json')); obs30, wave_med = _q['obs_max30'], _q['wave_median']
svc, bg_off = ins['exit']['main_cap_10min'] / 10, float(hourly[(hourly.is_off == 1) & (hourly.aud_big3 < 500) & (hourly.hour == 21) & (hourly.valid == 1)].out_main.mean()) / 60
def sim_max30(f, N):
    t = np.arange(-40, 221); a_ = ((t >= -30) & (t < 0)).astype(float); x = t - 5 + 0.5; b_ = np.where(x > 0, x / 64 * np.exp(-x / 8), 0); c_ = ((t >= 30) & (t < 120)).astype(float)
    wv = (1 - f) / 3 * a_ / a_.sum() + f * b_ / b_.sum() + (1 - f) * 2 / 3 * c_ / c_.sum()
    mu = svc - bg_off; q = 0; served = []
    for d_ in N * wv:
        sv = min(mu, q + d_); q = q + d_ - sv; served.append(sv + bg_off)
    return max(np.convolve(served, np.ones(30), 'valid'))
Nmed = wave_med * float(cars_per_1000) / 1000          # the calibration days' own audience, not KSPO DOME alone
lo_, hi_ = 0.3, 0.8
for _ in range(30):
    mid = (lo_ + hi_) / 2
    lo_, hi_ = (mid, hi_) if sim_max30(mid, Nmed) < obs30 else (lo_, mid)
burst = round((lo_ + hi_) / 2, 2)
print('observed max 30-min exits', obs30, '-> calibrated burst share', burst)
queue = dict(service_per_min=round(ins['exit']['main_cap_10min'] / 10, 1), cars_per_1000=round(float(cars_per_1000), 1), burst=burst, obs_max30=obs30,
             gate_share={k: v for k, v in ins['post_show_exit_share'].items() if k in ('동2', '북2', '남2')},
             gate_cap_10={k: v for k, v in ins['gate_exit_capacity_10min'].items() if k in ('동2', '북2', '남2')},
             surge_minutes=ins['exit']['surge_minutes_median'], show_len_min=int(XC['weekend']['peak_median']) - 10,   # 25_exit_clear.py: observed exit peak (slot end) - 10
             clear_p50=int(XC['weekend']['clear_p50']), clear_p75=int(XC['weekend']['clear_p75']),
             clear_p50_wk=int(round(XC['weekday']['clear_p50'] / 10) * 10), clear_p75_wk=int(round(XC['weekday']['clear_p75'] / 10) * 10),
             clear_n=XC['weekend']['n_days'], clear_n_wk=XC['weekday']['n_days'], calib_cars=round(Nmed), calib_wave=round(wave_med),
             bg_per_min={'off': round(float(hourly[(hourly.is_off == 1) & (hourly.aud_big3 < 500) & (hourly.hour == 21) & (hourly.valid == 1)].out_main.mean()) / 60, 1),
                         'wk': round(float(hourly[(hourly.is_off == 0) & (hourly.aud_big3 < 500) & (hourly.hour == 21) & (hourly.valid == 1)].out_main.mean()) / 60, 1)})

# ---- holdout and history
ho = pd.read_pickle(D + 'holdout_final.pkl')
hd = ho.groupby('date').agg(y=('occ_main', 'max'), p=('p', 'max'), hi=('hi', 'max'), b=('b1', 'max'), ev=('aud_big3', 'first')).reset_index()
hold = dict(d=[x.strftime('%Y-%m-%d') for x in hd.date], y=hd.y.round(0).astype(int).tolist(), p=hd.p.round(0).astype(int).tolist(),
            hi=hd.hi.round(0).astype(int).tolist(), b=hd.b.round(0).astype(int).tolist(), ev=(hd.ev >= 5000).astype(int).tolist())
wk = ho[(ho.date >= '2025-11-03') & (ho.date <= '2025-11-16')].sort_values('ts')
hold_week = dict(t=[x.strftime('%m-%d %H') for x in wk.ts], y=wk.occ_main.round(0).astype(int).tolist(), p=wk.p.round(0).astype(int).tolist(),
                 lo=wk.lo.round(0).astype(int).tolist(), hi=wk.hi.round(0).astype(int).tolist(), b=wk.b1.round(0).astype(int).tolist())
rep = json.load(open(D + 'holdout_report.json'))
hist = hourly[hourly.valid == 1].groupby('date').occ_main.max().reset_index()
history = dict(d=[x.strftime('%Y-%m-%d') for x in hist.date], pk=hist.occ_main.round(0).astype(int).tolist())

# ---- live check + agency comparison
# (the 2026-11-06 live check and the 11-07/08 comparison are kept from the previous build once those dates leave the window)
prev = json.load(open(D + 'app_data.json', encoding='utf-8')) if os.path.exists(D + 'app_data.json') else {}
live = pd.read_csv(D + 'live_api_log.csv') if os.path.exists(D + 'live_api_log.csv') else pd.DataFrame(columns=['ts_kst', 'source', 'parking_sum'])
live = live[(live.source == 'parkingorsApi') & live.ts_kst.astype(str).str.startswith('2026-11-06')]
has_live = (fut.date == '2026-11-06').any() and len(live) > 0 and os.path.exists('./backup_pre_jury/future_forecast.pkl')
f6 = fut[fut.date == '2026-11-06'].set_index('hour')
live_rows = [] if has_live else prev.get('live', [])
if has_live:
    old = pd.read_pickle('./backup_pre_jury/future_forecast.pkl')
    o6 = old[old.date == '2026-11-06'].set_index('hour')
for r in (live.itertuples() if has_live else []):
    hh = int(r.ts_kst[11:13])
    live_rows.append(dict(t=r.ts_kst[11:16], api=int(r.parking_sum), p=int(round(f6.p[hh])), lo=int(round(f6.lo[hh])), hi=int(round(f6.hi[hh])),
                          p0=int(round(o6.p[hh])), lo0=int(round(o6.lo[hh])), hi0=int(round(o6.hi[hh]))))
ag = pd.read_csv(D + 'agency_forecast_20261107_08.csv') if os.path.exists(D + 'agency_forecast_20261107_08.csv') else pd.DataFrame(columns=['slot', 'agency_pct'])
has_ag = len(ag) > 0 and fut.date.isin(pd.to_datetime(ag.slot.str[:10])).all()
agency = [] if has_ag else prev.get('agency', [])
for r in (ag.itertuples() if has_ag else []):
    dd, hh = r.slot[:10], int(r.slot[11:13])
    row = fut[(fut.date == dd) & (fut.hour == hh)].iloc[0]
    agency.append(dict(slot=r.slot, ag=int(r.agency_pct), p=int(round(row.p)), pct=int(round(row.p / CAP * 100))))

meta_now = json.load(open(D + 'final_meta.json'))
fieldcheck = json.load(open(D + 'field_check.json', encoding='utf-8')) if os.path.exists(D + 'field_check.json') else None   # 22_field_check.py
live2 = json.load(open(D + 'live_prereg.json', encoding='utf-8')) if os.path.exists(D + 'live_prereg.json') else []   # published forecast vs live API
live2_summary = json.load(open(D + 'live_prereg_summary.json', encoding='utf-8')) if os.path.exists(D + 'live_prereg_summary.json') else None   # metrics, by day and group, sentences, noon update
app = dict(cap=CAP, generated=TODAY, first_pub='2026-11-15', view=os.environ.get('OLPARK_VIEW', 'all'), data_end=meta_now.get('data_end'), growth=round(meta_now.get('growth_now', 1.0), 3), sched_horizon=sched_horizon, days=days, resp_aud=resp_meta, queue=queue,
           hold=hold, hold_week=hold_week, metrics=rep, history=history, live=live_rows, agency=agency, fieldcheck=fieldcheck, live2=live2, live2_summary=live2_summary,
           same_alarm=json.load(open(D + 'robustness.json'))['same_alarm_days'], demo=json.load(open(D + 'robustness.json'))['full_demo'],
           insights=dict(arrival=dict(r50=ins['arrival']['rel_50'], r75=ins['arrival']['rel_75'], r90=ins['arrival']['rel_90'], t90=ins['arrival']['t90_peak_rel_median']),
                         stay=ins['stay'], lift=dict(ev=ins['lift']['peak_event_mean'], no=ins['lift']['peak_noevent_mean']),
                         daytype=ins['daytype_peak'], weather=ins['weather_off']))
s = json.dumps(app, ensure_ascii=False, separators=(',', ':'))
open(D + 'app_data.json', 'w', encoding='utf-8').write(s)
print('app_data.json', len(s) // 1024, 'KB; days', len(days), 'horizon', sched_horizon, 'queue', queue)
print('live', live_rows)
print('agency', agency[:6])
