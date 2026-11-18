"""Daily event features per venue: show flag, first start time, estimated audience.

Audience = rental requested attendance (REQST, known before the show; fallback EXCCLC) split over the scheduled
show-days of that venue inside the rental period, weighted by number of shows that day.
Rentals with no scheduled show inside (sports, public events) are spread over rental days.
Scheduled shows with no rental match get the venue's historical median per show-day.
"""
import pandas as pd, numpy as np, datetime as dt, re, os

SETTLED = os.environ.get('OLPARK_ATT') == 'settled'   # settled audience known after the show: descriptive analysis only (13, 14, 17, 18)
SUF = '_settled' if SETTLED else ''

D = './data/'
rent = pd.read_pickle(D + 'rent.pkl')
ev = pd.read_csv(D + 'events_parsed.csv', dtype={'plcCd': str})
sd = pd.read_csv(D + 'show_days.csv', parse_dates=['date'])

def venue_from_sched(plc, code):
    p = str(plc)
    if '88잔디' in p or '수변무대' in p or '88호수' in p or '광장' in p:
        return 'OUTDOOR'
    if 'KSPO' in p.upper() or '체조' in p:
        return 'KSPO'
    if '핸드볼' in p or 'LINK' in p.upper():
        return 'HANDBALL'
    if '뮤즈' in p:
        return 'MUSE'
    if '아트홀' in p and 'K-' not in p:
        return 'ARTHALL'
    if '올림픽홀' in p:
        return 'OLHALL'
    return {'02': 'KSPO', '03': 'HANDBALL', '01': 'OLHALL', '04': 'ARTHALL', '07': 'MUSE', '06': 'OUTDOOR'}.get(str(code).strip(), 'OTHER')

ev['venue'] = [venue_from_sched(p, c) for p, c in zip(ev['plc'], ev['plcCd'])]
sd = sd.merge(ev[['evtSeq', 'venue', 'title', 'ndTm']], on='evtSeq', how='left')

RENT_VENUE = {'03': 'KSPO', '05': 'HANDBALL', '09': 'OLHALL', '07': 'ARTHALL', '10': 'MUSE',
              '21': 'OUTDOOR', '33': 'OUTDOOR', '23': 'OUTDOOR', '19': 'OUTDOOR', '27': 'OUTDOOR', '25': 'PLAZA'}
rent['venue'] = rent['RENTLSPCE_HOUR_DC'].str.strip().map(RENT_VENUE).fillna('MINOR')
rent['settled'] = rent['EXCCLC_EVENT_NMPR_CO'].fillna(0) > 0
if SETTLED:
    rent['att'] = rent['EXCCLC_EVENT_NMPR_CO'].where(rent['settled'], rent['REQST_EVENT_NMPR_CO'])
else:
    rent['att'] = rent['REQST_EVENT_NMPR_CO'].where(rent['REQST_EVENT_NMPR_CO'].fillna(0) > 0, rent['EXCCLC_EVENT_NMPR_CO'])   # 신청 인원 우선: 공연 전에 알 수 있는 값
rent = rent[(rent['e'] >= '2023-12-01') & rent['venue'].isin(['KSPO', 'HANDBALL', 'OLHALL', 'ARTHALL', 'MUSE', 'OUTDOOR', 'PLAZA'])].copy()
tentative = rent[~rent['settled'] & rent['att'].fillna(0).gt(0)].copy()   # holds without settlement: used only for name-matched shows
rent = rent[rent['settled']].copy()
# 만남의광장(PLAZA) rentals mirror arena concerts (MD booths/queues): drop when another venue has the same event name
names_main = set(rent.loc[rent.venue != 'PLAZA', 'EVENT_NM'].str.replace(r'\s+', '', regex=True))
rent = rent[~((rent.venue == 'PLAZA') & rent['EVENT_NM'].str.replace(r'\s+', '', regex=True).isin(names_main))]
rent.loc[rent.venue == 'PLAZA', 'venue'] = 'OUTDOOR'
# outdoor festivals often book several lawn/stage areas with the same attendance: keep one per event name
rent = rent.sort_values('att', ascending=False).drop_duplicates(subset=['venue', 'EVENT_NM', 'RENTLSPCE_BEGIN_DE'])

day_rows = []
used_show = set()
for r in rent.itertuples():
    days = pd.date_range(r.b, r.e, freq='D')
    s = sd[(sd.venue == r.venue) & sd.date.between(r.b, r.e)]
    if len(s):
        w = s['n_shows'].clip(upper=3).astype(float)
        share = r.att * w / w.sum()
        for (idx, row), a in zip(s.iterrows(), share):
            day_rows.append(dict(date=row.date, venue=r.venue, aud=a, start=row.first_start, n_shows=row.n_shows,
                                 src='rent+sched', name=r.EVENT_NM, evtSeq=row.evtSeq))
            used_show.add(idx)
    else:
        # unscheduled rental (sport/public/other): spread over rental days, start unknown
        n = len(days)
        for d in days:
            day_rows.append(dict(date=d, venue=r.venue, aud=r.att / n, start=np.nan, n_shows=1,
                                 src='rent_only', name=r.EVENT_NM, evtSeq=np.nan))
ed = pd.DataFrame(day_rows)
# per show-day medians by venue (for unmatched shows and for the forecast period)
med = ed[ed.src == 'rent+sched'].groupby('venue')['aud'].median()
print('median audience per show-day by venue:\n', med.round(0))
cap = {'KSPO': 15000, 'HANDBALL': 6000, 'OLHALL': 3000, 'ARTHALL': 1200, 'MUSE': 300, 'OUTDOOR': 30000}
def toks(s):
    s = re.sub(r'\(가제\)|콘서트|단독|공연|CONCERT|TOUR|WORLD|IN SEOUL|서울|20\d\d', ' ', str(s).upper())
    return {t for t in re.findall(r'[A-Z]{2,}|[가-힣]{2,}', s)}

rest = sd[~sd.index.isin(used_show)]
fill = []
for idx, row in rest.iterrows():
    a, src = med.get(row.venue, 500), 'sched_only'
    cand = tentative[(tentative.venue == row.venue) & (tentative.b <= row.date) & (tentative.e >= row.date)]
    if len(cand):
        tt = toks(row.title)
        best = max(cand.itertuples(), key=lambda c: len(tt & toks(c.EVENT_NM)))
        if len(tt & toks(best.EVENT_NM)) >= 1:
            n_days = sd[(sd.evtSeq == row.evtSeq)].shape[0] or 1
            a, src = best.att / n_days, 'sched+request'
    fill.append(dict(date=row.date, venue=row.venue, aud=a, start=row.first_start,
                     n_shows=row.n_shows, src=src, name=row.title, evtSeq=row.evtSeq))
ed = pd.concat([ed, pd.DataFrame(fill)], ignore_index=True)
ed.to_csv(D + f'event_days_long{SUF}.csv', index=False)

# wide daily table
V = ['KSPO', 'HANDBALL', 'OLHALL', 'ARTHALL', 'MUSE', 'OUTDOOR']
g = ed[ed.venue.isin(V)].groupby(['date', 'venue']).agg(aud=('aud', 'sum'), start=('start', 'min'), n_shows=('n_shows', 'max')).reset_index()
# physical cap per venue-day (seats x shows, at most 2 full houses a day)
g['aud'] = [min(a, cap[v] * min(max(n, 1), 2)) for a, v, n in zip(g.aud, g.venue, g.n_shows.fillna(1))]
wide = g.pivot(index='date', columns='venue')
wide.columns = [f'{a}_{b}' for a, b in wide.columns]
wide = wide.reset_index()
for v in V:
    for c in ['aud', 'start', 'n_shows']:
        if f'{c}_{v}' not in wide:
            wide[f'{c}_{v}'] = np.nan
    wide[f'aud_{v}'] = wide[f'aud_{v}'].fillna(0)
wide['aud_total'] = wide[[f'aud_{v}' for v in V]].sum(axis=1)
wide['aud_big3'] = wide[['aud_KSPO', 'aud_HANDBALL', 'aud_OLHALL']].sum(axis=1)
wide.to_csv(D + f'event_days_wide{SUF}.csv', index=False)
print(ed.groupby(['venue', 'src']).size().unstack(fill_value=0))
print('days with any event 2024-01-08..2025-12-30:',
      wide[(wide.date >= '2024-01-08') & (wide.date <= '2025-12-30')].shape[0])
chk = wide.set_index('date').loc[['2024-09-29', '2025-04-06', '2024-09-07', '2025-11-01'], ['aud_KSPO', 'start_KSPO', 'aud_HANDBALL', 'start_HANDBALL', 'aud_OLHALL', 'start_OLHALL', 'aud_OUTDOOR', 'aud_total']]
print(chk.round(0).to_string())
