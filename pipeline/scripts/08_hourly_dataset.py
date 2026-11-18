"""Hourly modelling table for 2024-01-08..2025-12-30 plus a future frame for the forecast horizon."""
import duckdb, pandas as pd, numpy as np, datetime as dt, os

D = './data/'
SUF = '_settled' if os.environ.get('OLPARK_ATT') == 'settled' else ''   # settled-audience copy for descriptive analysis
HOL = {
    # 2024
    '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-03-01', '2024-04-10', '2024-05-05',
    '2024-05-06', '2024-05-15', '2024-06-06', '2024-08-15', '2024-09-16', '2024-09-17', '2024-09-18', '2024-10-01',
    '2024-10-03', '2024-10-09', '2024-12-25',
    # 2025
    '2025-01-01', '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-03-01', '2025-03-03', '2025-05-05',
    '2025-05-06', '2025-06-03', '2025-06-06', '2025-08-15', '2025-10-03', '2025-10-05', '2025-10-06', '2025-10-07',
    '2025-10-08', '2025-10-09', '2025-12-25',
    # 2026
    '2026-01-01', '2026-02-16', '2026-02-17', '2026-02-18', '2026-03-01', '2026-03-02', '2026-05-01', '2026-05-05',
    '2026-05-24', '2026-05-25', '2026-06-03', '2026-06-06', '2026-07-17', '2026-08-15', '2026-08-17', '2026-09-24',
    '2026-09-25', '2026-09-26', '2026-10-03', '2026-10-05', '2026-10-09', '2026-12-25',
    # 2027 (Jan-Mar)
    '2027-01-01', '2027-02-06', '2027-02-07', '2027-02-08', '2027-02-09', '2027-03-01',
}
HOL = {pd.Timestamp(d) for d in HOL}
V = ['KSPO', 'HANDBALL', 'OLHALL', 'ARTHALL', 'MUSE', 'OUTDOOR']


def calendar_frame(start, end):
    hrs = pd.date_range(start, end, freq='h')
    f = pd.DataFrame({'ts': hrs})
    f['date'] = f.ts.dt.normalize()
    f['hour'] = f.ts.dt.hour
    f['dow'] = f.ts.dt.dayofweek
    f['month'] = f.ts.dt.month
    f['doy'] = f.ts.dt.dayofyear
    f['doy_sin'] = np.sin(2 * np.pi * f.doy / 365.25)
    f['doy_cos'] = np.cos(2 * np.pi * f.doy / 365.25)
    f['is_hol'] = f.date.isin(HOL).astype(int)
    f['is_off'] = ((f.dow >= 5) | (f.is_hol == 1)).astype(int)
    nxt = f.date + pd.Timedelta(days=1)
    prv = f.date - pd.Timedelta(days=1)
    f['next_off'] = ((nxt.dt.dayofweek >= 5) | nxt.isin(HOL)).astype(int)
    f['prev_off'] = ((prv.dt.dayofweek >= 5) | prv.isin(HOL)).astype(int)
    md = f.ts.dt.month * 100 + f.ts.dt.day
    f['school_vac'] = (((md >= 721) & (md <= 820)) | (md >= 1224) | (md <= 228)).astype(int)
    return f


def add_events(f):
    w = pd.read_csv(D + f'event_days_wide{SUF}.csv', parse_dates=['date'])
    f = f.merge(w, on='date', how='left')
    mid = f.hour * 60 + 30
    for v in V:
        f[f'aud_{v}'] = f[f'aud_{v}'].fillna(0)
        f[f'laud_{v}'] = np.log1p(f[f'aud_{v}'])
        f[f'rel_{v}'] = np.where(f[f'aud_{v}'] > 0, mid - f[f'start_{v}'], np.nan)   # minutes from first show start
        # crude presence kernel: arrival ramps over 3h before start, stays ~3h after start, drains for 2h
        r = f[f'rel_{v}']
        k = np.where(r.isna(), 0.0,
             np.where(r < -180, 0.0,
             np.where(r < 0, (r + 180) / 180,
             np.where(r < 180, 1.0,
             np.where(r < 300, (300 - r) / 120, 0.0)))))
        f[f'press_{v}'] = k * f[f'aud_{v}']
    for c in ['aud_total', 'aud_big3']:
        f[c] = f[c].fillna(0)
    f['laud_total'] = np.log1p(f['aud_total'])
    f['press_total'] = f[[f'press_{v}' for v in V]].sum(axis=1)
    f['n_venues'] = (f[[f'aud_{v}' for v in V]] > 0).sum(axis=1)
    return f


def add_weather(f, wx):
    f = f.merge(wx, on='date', how='left')
    f['rain_mm'] = f['rain_mm'].fillna(0)
    f['rainy'] = (f['rain_mm'] >= 1).astype(int)
    f['heavy_rain'] = (f['rain_mm'] >= 10).astype(int)
    return f


if __name__ == '__main__':
    con = duckdb.connect(D + 'olpark.duckdb', read_only=True)
    tgt = con.execute("""
      SELECT date_trunc('hour', slot) ts, max(occ_peak) occ_main, sum(n_in) in_main, sum(n_out) out_main
      FROM slot_main GROUP BY 1""").df()
    tga = con.execute("""
      SELECT date_trunc('hour', slot) ts, max(occ_peak) occ_all FROM slot_all GROUP BY 1""").df()
    first, last = con.execute("SELECT min(slot), max(slot) FROM slot_main").fetchone()   # follows the exported data
    f = calendar_frame(f'{first:%Y-%m-%d} 00:00', f'{last:%Y-%m-%d} 23:00')
    f = f.merge(tgt, on='ts', how='left').merge(tga, on='ts', how='left')
    wx = pd.read_csv(D + 'weather_seoul_daily.csv', parse_dates=['date'])
    f = add_events(f)
    f = add_weather(f, wx)
    # flag hours we should not train/evaluate on: data gaps and the day after a gap (occupancy carry-over unknown)
    gaps = pd.to_datetime(['2024-02-01', '2024-12-31', '2025-07-31', '2025-08-02'])
    bad = set()
    for g in gaps:
        bad |= {g, g + pd.Timedelta(days=1)}
    bad |= set(pd.date_range('2024-02-01', '2024-03-01'))
    bad |= set(pd.date_range('2024-01-08', '2024-01-09'))
    f['valid'] = (~f.date.isin(bad)).astype(int)
    f.to_pickle(D + f'hourly{SUF}.pkl')
    print(f.shape, f.valid.mean().round(3))
    print(f[['occ_main', 'occ_all', 'in_main', 'out_main']].describe().round(0).to_string())
