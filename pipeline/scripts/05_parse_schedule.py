"""Parse KSPO&CO show schedule free-text times into (event, date, start_time) rows."""
import json, re, datetime as dt
import pandas as pd

events = json.load(open('./data/events_ksponco.json', encoding='utf-8'))
# shows announced by the organisers but not yet in the public show list (kept apart and marked src=manual)
import os
_man = './data/events_manual.json'
if os.path.exists(_man):
    def _listed(m):        # the same venue already has a public entry over these dates
        return any(str(e.get('plcCd', '')).strip() == str(m.get('plcCd', '')).strip() and str(e.get('beginDt', ''))[:10] <= m['endDt'] and m['beginDt'] <= str(e.get('endDt', ''))[:10] for e in events)
    events += [m for m in json.load(open(_man, encoding='utf-8')) if not _listed(m)]   # a manual entry gives way once the show is in the public list
WD = '월화수목금토일'

def norm(s):
    s = s.replace('：', ':').replace('；', ';').replace('（', '(').replace('）', ')')
    s = re.sub(r'(\d)\s*;\s*(\d)', r'\1:\2', s)          # 19;00 -> 19:00
    s = s.replace('시간', 'H')                              # avoid '시' collisions
    for w in '월화수목금토일':
        s = s.replace(w + '요일', w)
    s = s.replace('평일', '월화수목금').replace('주말', '토일')
    s = re.sub(r'([월화수목금토일])\s*/\s*(?=[월화수목금토일])', r'\1', s)    # 토/일 -> 토일
    return s

def no_show_days(s):
    out = set()
    for m in re.finditer(r'([월화수목금토일][월화수목금토일,\s]*)\s*(?:요일)?\s*공연\s*없음', s):
        out |= {ch for ch in m.group(1) if ch in '월화수목금토일'}
    return out

TIME_RE = re.compile(
    r'(?P<ampm>오전|오후|저녁|낮|밤)?\s*'
    r'(?:(?P<h1>\d{1,2})\s*:\s*(?P<m1>\d{2})(?::\d{2})?\s*(?P<pm1>PM|AM|pm|am)?'
    r'|(?P<h2>\d{1,2})\s*(?P<pm2>PM|AM|pm|am)'
    r'|(?P<h3>\d{1,2})\s*시(?:\s*(?P<m3>\d{1,2})\s*분)?)'
)

AMPM_BARE = re.compile(r'(오전|오후|저녁|밤)\s*(\d{1,2})(?!\s*(?:\d|:|시|AM|PM|am|pm))')

def times_in(s):
    out = []
    for m in AMPM_BARE.finditer(s):          # "오후 5" with the hour mark cut off
        h = int(m.group(2))
        if m.group(1) != '오전' and h < 12: h += 12
        if 0 <= h <= 23: out.append(h * 60)
    for m in TIME_RE.finditer(s):
        if m.group('h1'):
            h, mi, suf = int(m.group('h1')), int(m.group('m1')), (m.group('pm1') or '').upper()
        elif m.group('h2'):
            h, mi, suf = int(m.group('h2')), 0, (m.group('pm2') or '').upper()
        else:
            h, mi, suf = int(m.group('h3')), int(m.group('m3') or 0), ''
        ap = m.group('ampm')
        if suf == 'PM' and h < 12: h += 12
        if ap in ('오후', '저녁', '밤') and h < 12: h += 12
        if not suf and not ap and m.group('h3') and 1 <= h <= 9:
            h += 12   # bare "6시" for a concert means 18시
        if 0 <= h <= 23 and 0 <= mi < 60:
            out.append(h * 60 + mi)
    return out

def daterange(b, e):
    d = b
    while d <= e:
        yield d
        d += dt.timedelta(days=1)

def weekday_schedule(tm, days):
    """weekday groups: "(토) 18:00 / (일) 17:00", "화~금 11:00", "토일 18:00" -> {date: [mins]}"""
    wk_re = re.compile(r'\(?(?P<w>[월화수목금토일](?:\s*[,·~]?\s*\)?\s*\(?[월화수목금토일])*)\s*\)?(?=\s*(?:요일)?\s*[:)]?\s*(?:\(\d회\))?\s*(?:오전|오후|저녁|\d))')
    marks = []
    for m in wk_re.finditer(tm):
        w = m.group('w')
        if '~' in w:
            a, z = w[0], w[-1]
            ws = WD[WD.index(a): WD.index(z) + 1]
        else:
            ws = ''.join(ch for ch in w if ch in WD)
        marks.append((m.start(), m.end(), ws))
    wmap = {}
    for i, (s0, s1, ws) in enumerate(marks):
        seg = tm[s1: marks[i + 1][0] if i + 1 < len(marks) else len(tm)]
        seg = re.split(r'\d{1,2}\s*[./월]\s*\d{1,2}', seg)[0]     # stop at an explicit date override
        t = times_in(seg)
        for ch in ws:
            if t:
                wmap.setdefault(ch, []).extend(t)
    off = no_show_days(tm)
    res = {}
    for c in days:
        w = WD[c.weekday()]
        if w in wmap and w not in off:
            res[c] = list(wmap[w])
    return res

def parse_event(ev):
    b = dt.date.fromisoformat(ev['beginDt']); e = dt.date.fromisoformat(ev['endDt'])
    tm = norm(ev['tm'] or '')
    days = list(daterange(b, e))
    res = {}      # date -> list of start minutes
    method = None
    # long runs: weekday pattern is the base schedule, explicit dates override
    if len(days) > 7:
        res = weekday_schedule(tm, days)
        if res:
            method = 'weekday'
    # 1) explicit month/day mentions: "11월 13일", "8.3(토)", "2024년 11월 1일(금)", "12월 27일(금)"
    md_re = re.compile(r'(?:(?P<y>\d{4})\s*[년.]\s*)?(?P<mo>\d{1,2})\s*(?:월|\.|,)\s*(?P<d>\d{1,2})\s*일?\s*(?:\(?\s*(?P<w>[월화수목금토일])\s*\)?)?')
    marks = []
    for m in md_re.finditer(tm):
        mo, d_ = int(m.group('mo')), int(m.group('d'))
        if not (1 <= mo <= 12 and 1 <= d_ <= 31):
            continue
        # skip things that are actually times like "7.5" inside other context: require date within event range (+/- year)
        cand = None
        for y in (b.year, e.year):
            try:
                c = dt.date(y, mo, d_)
            except ValueError:
                continue
            if b - dt.timedelta(days=1) <= c <= e + dt.timedelta(days=1):
                cand = c
        if cand is None and m.group('w'):
            # month typo (e.g. "1월 15일 일" inside a November run): match day number + weekday
            for c in days:
                if c.day == d_ and WD[c.weekday()] == m.group('w'):
                    cand = c
        if cand:
            marks.append((m.start(), m.end(), cand))
    if marks and len(days) > 7 and res:
        for i, (s0, s1, d0) in enumerate(marks):
            seg = tm[s1: marks[i + 1][0] if i + 1 < len(marks) else len(tm)]
            t = times_in(seg)
            if t:
                res[d0] = t           # override the base pattern for that date
        marks = []
    elif marks:
        res = {}
    # day-only with weekday: "26일(금) 8PM", "30(금) 17:00" -- merged with month/day marks when both styles appear
    dd_re = re.compile(r'(?<![\d.월])(?P<d>\d{1,2})\s*일?\s*\(\s*(?P<w>[월화수목금토일])\s*\)')
    dmarks = []
    if not (len(days) > 7 and method == 'weekday'):
        for m in dd_re.finditer(tm):
            if any(s0 <= m.start() < s1 for s0, s1, _ in marks):
                continue                                  # already covered by a month/day mark
            d_ = int(m.group('d'))
            for c in days:
                if c.day == d_ and WD[c.weekday()] == m.group('w'):
                    dmarks.append((m.start(), m.end(), c))
    allm = sorted(marks + dmarks)
    if allm:
        for i, (s0, s1, d0) in enumerate(allm):
            seg = tm[s1: allm[i + 1][0] if i + 1 < len(allm) else len(tm)]
            t = times_in(seg)
            if t:
                res.setdefault(d0, []).extend(t)
        if res:
            method = 'date' if marks else 'daynum'
    # date marks without their own times ("11월 14일(토) ~ 11월 15(일)  토 오후 6시 / 일 오후 4시"):
    # when a weekday line covers every day of a short run, it wins over the date marks
    if res and len(days) <= 7:
        wres = weekday_schedule(re.sub(r'\d{1,2}\s*일?\s*\(\s*[월화수목금토일]\s*\)', ' # ', tm), days)   # "08일(일)" is a date, not a weekday group
        if wres and set(wres) == set(days) and set(res) != set(days):
            res, method = wres, 'weekday'
    # 3) weekday groups: "(토) 18:00 / (일) 17:00", "금 오후 7시 / 토 오후 6시", "(토)(일) 14:00, 19:00", "토일 18:00"
    if not res:
        res = weekday_schedule(tm, days)
        if res:
            method = 'weekday'
    # 4) no time text at all for a short run: assume typical start (festival noon, concerts by weekday)
    if not res and not times_in(tm) and len(days) <= 4:
        outdoor = any(k in str(ev.get('plc', '')) for k in ('88잔디', '수변', '88호수', '광장')) or str(ev.get('plcCd', '')).strip() == '06' or '페스티벌' in str(ev.get('title', ''))
        for c in days:
            res[c] = [720] if outdoor else [{4: 1170, 5: 1080, 6: 1020}.get(c.weekday(), 1170)]
        method = 'default'
    # 5) plain times apply to every day in range (only for short runs)
    if not res:
        t = times_in(tm)
        if t and len(days) <= 4:
            for c in days:
                res[c] = list(t)
            method = 'plain'
        elif t:
            method = 'longrun'
    rows = []
    for d0, ts in sorted(res.items()):
        ts = sorted(set(ts))
        rows.append(dict(evtSeq=ev['evtSeq'], date=d0, first_start=ts[0], last_start=ts[-1], n_shows=len(ts)))
    return rows, method

all_rows, meth = [], {}
for ev in events:
    rows, m = parse_event(ev)
    meth[ev['evtSeq']] = m
    all_rows.extend(rows)
df = pd.DataFrame(all_rows)
evdf = pd.DataFrame(events)
evdf['method'] = evdf['evtSeq'].map(meth)
evdf['plcCd'] = evdf['plcCd'].str.strip()
evdf['run_days'] = (pd.to_datetime(evdf['endDt']) - pd.to_datetime(evdf['beginDt'])).dt.days + 1
print(evdf['method'].value_counts(dropna=False))
un = evdf[evdf['method'].isna() | (evdf['method'] == 'longrun')]
print('unparsed/longrun:', len(un))
print(un[['evtSeq', 'plcCd', 'beginDt', 'endDt', 'run_days', 'tm']].to_string(max_colwidth=90))
df.to_csv('./data/show_days.csv', index=False)
evdf.to_csv('./data/events_parsed.csv', index=False)
