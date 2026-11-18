"""Daily forecast-only batch: reuse the saved models of 12_final_model.py, refresh the forecast and rebuild the web file.

Usage:  python 20_daily_forecast.py [start YYYY-MM-DD] [days] [--fallback]
        --fallback: backup mode without the model, simple weekday-and-hour average of the last 52 weeks
        (environment variable OLPARK_TODAY sets the 'generated' date shown on the page; default: today in Korea)

- Re-reads events, rentals and weather, so run the event and rental steps first when new schedules arrive
  (00_fetch_events.py, 05_parse_schedule.py, 07_event_features.py).
- Recent level: if the parking entry and exit data reach within 28 days of the start date, the real last weeks are used
  (window [d-35, d-7], frozen at the newest available window for later days). Otherwise the same window a year earlier
  is used, multiplied by the year-on-year growth of daytime occupancy over 13 weeks (public demo condition; see growth_end).
- Weather: average of the same calendar day over the two previous years, observed values where they exist.
- Writes data/future_forecast.pkl, then runs 14_app_data.py and 15_build_app.py.
- Retrain monthly with 12_final_model.py.
"""
import os, sys, json, runpy, pandas as pd, numpy as np, lightgbm as lgb
from importlib.machinery import SourceFileLoader
S = os.path.dirname(os.path.abspath(__file__)) + '/'
exec(open(S + '10_model_v2.py', encoding='utf-8').read().split("res = []")[0])     # data prep: allf, add_recency, D
hd = SourceFileLoader('hd', S + '08_hourly_dataset.py').load_module()

now_kst = pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).normalize()
args = [a for a in sys.argv[1:] if not a.startswith('--')]
FALLBACK = '--fallback' in sys.argv
start = pd.Timestamp(args[0]) if len(args) > 0 else now_kst
days = int(args[1]) if len(args) > 1 else 120
end = start + pd.Timedelta(days=days - 1)
os.environ.setdefault('OLPARK_TODAY', now_kst.strftime('%Y-%m-%d'))

meta = json.load(open(D + 'final_meta.json'))
FF, OFF = meta['FF'], meta['OFF']
models = [] if FALLBACK else [lgb.Booster(model_file=D + f'final_model_{i}.txt') for i in range(3)]   # backup mode needs no model files
ks = sorted(int(k) for k in meta['centres'])
CEN, Q10, Q90 = (np.array([meta[n][str(k)] for k in ks]) for n in ('centres', 'q10', 'q90'))

# ---- calendar, events, weather
fut = hd.add_events(hd.calendar_frame(f'{start:%Y-%m-%d} 00:00', f'{end:%Y-%m-%d} 23:00'))
WCOLS = ['avg_temp', 'max_temp', 'min_temp', 'avg_cloud']
wx = pd.read_csv(D + 'weather_seoul_daily.csv', parse_dates=['date'])
clim = wx[wx.date.dt.year.isin([start.year - 2, start.year - 1])].assign(md=lambda x: x.date.dt.strftime('%m-%d')).groupby('md')[WCOLS].mean()
fut['md'] = fut.date.dt.strftime('%m-%d')
fut = fut.merge(clim, left_on='md', right_index=True, how='left')
known = wx.set_index('date')
for c in WCOLS:
    fut[c] = [known[c].get(d, np.nan) if d in known.index and pd.notna(known[c].get(d, np.nan)) else v for d, v in zip(fut.date, fut[c])]
fut['rain_mm'] = 0.0; fut['rainy'] = 0; fut['heavy_rain'] = 0
fut['post_sys'] = 1
fut['daytype'] = np.where(fut.is_hol == 1, 6, fut.dow)

# ---- recent level
hist = allf[['date', 'hour', 'dow', 'is_hol', 'occ_main']].copy()
data_end = allf.date.max()
dayt = allf[allf.hour.between(10, 20)]
# the parking system was replaced on 2024-09-12 and late-2024 records are thin (north gate 2 entries about 61% of a year later).
# When the year-earlier half of the 13-week growth window falls in that stretch, the window is moved back to end a year after
# the change (2025-09-11); otherwise it ends at the last data day. OLPARK_GROWTH_END overrides the end date.
SYS_CHANGE, SYS_SETTLED = pd.Timestamp('2024-09-12'), pd.Timestamp('2024-12-31')
def growth_end(e, weeks=13):
    e = pd.Timestamp(e); a1 = e - pd.Timedelta(days=364); a0 = a1 - pd.Timedelta(weeks=weeks)
    return e if (a1 < SYS_CHANGE or a0 > SYS_SETTLED) else min(e, SYS_CHANGE + pd.Timedelta(days=364))
def growth(e, weeks=13):
    e = growth_end(e)
    w0, ly = pd.Timedelta(weeks=weeks), pd.Timedelta(days=364)
    a = dayt[(dayt.date > e - w0) & (dayt.date <= e)].occ_main.mean()
    b = dayt[(dayt.date > e - w0 - ly) & (dayt.date <= e - ly)].occ_main.mean()
    return float(a / b)
def recency(dates_shift=0, freeze=None, g=1.0, rows=None):
    fr = fut if rows is None else fut[rows]
    px = fr[['date', 'hour', 'dow', 'is_hol']].copy()
    px['date'] = px.date - pd.Timedelta(days=dates_shift)
    if freeze is not None:
        px['date'] = px.date.where(px.date <= freeze, freeze)
    t = add_recency(px, hist)
    return t.rec_hour.values * g, t.rec_dpk.values * g

if start - data_end <= pd.Timedelta(days=28):   # weekly or monthly exports keep the real recent level (validation: 28-day-old level, error 139)
    mode, g = 'recent data', 1.0
    fut['rec_hour'], fut['rec_dpk'] = recency(freeze=data_end + pd.Timedelta(days=7))
else:
    mode, g = 'a year earlier x growth', growth(data_end)
    if os.path.exists(D + 'growth_check.json'):      # growth of stays up to one day (24_growth_check.py)
        g = json.load(open(D + 'growth_check.json'))['growth_demo']
    fut['rec_hour'], fut['rec_dpk'] = recency(364, g=g)
    for k in (2, 3, 4):                              # a year earlier is beyond the data -> go back whole years
        miss = fut.rec_hour.isna().values
        if not miss.any():
            break
        rh, rd = recency(364 * k, g=g, rows=miss)
        fut.loc[miss, 'rec_hour'], fut.loc[miss, 'rec_dpk'] = rh, rd
print(f'forecast {start:%Y-%m-%d}..{end:%Y-%m-%d}, data end {data_end:%Y-%m-%d}, recent level: {mode}, growth {g:.3f}, missing {int(fut.rec_hour.isna().sum())}')

# ---- predict, 80% band, save, rebuild the page
if FALLBACK:   # backup mode: weekday-and-hour average of the last 52 weeks of data (x growth when the data are old)
    last = allf[allf.date > data_end - pd.Timedelta(weeks=52)]
    b1 = last.groupby(['daytype', 'hour']).occ_main.mean()
    fut['p'] = fut.set_index(['daytype', 'hour']).index.map(b1).values * g
    print('backup mode: weekday average')
else:
    r = np.mean([m.predict(fut[FF]) for m in models], axis=0)
    fut['p'] = np.clip(r * (fut.rec_hour.values + OFF), 0, None)
fut['lo'] = np.clip((fut.p + OFF) * np.interp(fut.p, CEN, Q10) - OFF, 0, None)
fut['hi'] = (fut.p + OFF) * np.interp(fut.p, CEN, Q90) - OFF
fut.to_pickle(D + 'future_forecast.pkl')
runpy.run_path(S + '14_app_data.py', run_name='__main__')
runpy.run_path(S + '15_build_app.py', run_name='__main__')
