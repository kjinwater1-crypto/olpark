"""올림픽공원 홈페이지 주차 안내 페이지의 실시간 주차 대수와 시간대별 예상 이용률을 기록합니다.
GitHub Actions가 20분마다 실행합니다.   사용법: python record.py <기록 폴더>"""
import csv, datetime as dt, os, re, sys, urllib.parse, urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
URL = 'https://www.ksponco.or.kr/online/main/parking.do'
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')


def append(name, header, rows):
    path = os.path.join(OUT, name)
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def parse(html):
    html = re.sub(r'<!--.*?-->', ' ', html, flags=re.S).replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html))
    m = re.search(r'실시간\s*주차\s*대수[^:：]{0,40}[:：]\s*([\d,]+)\s*대\s*주차\s*중\s*/\s*([\d,]+)\s*대', text)
    if not m:
        raise ValueError('live count not found: ' + text[:200])
    n, cap = (int(g.replace(',', '')) for g in m.groups())
    slots = [(f'{d} {t}', int(p)) for d, t, p in re.findall(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})(?::\d{2})?\s*\|?\s*(\d{1,3})\s*%', text)]
    if not slots:                              # keep a short excerpt of the expected-utilisation part to check the page layout
        i = max(text.find('예상 이용률'), text.find('이용률'), 0)
        open(os.path.join(OUT, 'page_excerpt.txt'), 'w', encoding='utf-8').write(text[max(0, i - 1500):i + 2500])
    return n, cap, slots


if __name__ == '__main__':
    try:
        UA = {'User-Agent': 'Mozilla/5.0 (compatible; olpark-live-log; +https://github.com/kjinwater1-crypto/olpark)'}
        get = lambda data=None: urllib.request.urlopen(urllib.request.Request(URL, data=data, headers=UA), timeout=40).read().decode('utf-8', 'replace')
        html = get()
        n, cap, slots = parse(html)
        for d in re.findall(r'<option value="(\d{4}-\d{2}-\d{2})"', html)[1:]:   # the other weekend day(s) of the selector
            try:
                slots += parse(get(urllib.parse.urlencode({'selectdate': d}).encode()))[2]
            except Exception as e:
                append('errors.csv', ['ts_kst', 'error'], [[now, f'selectdate {d}: ' + repr(e)[:200]]])
        append('rt_log.csv', ['ts_kst', 'parking_now', 'cap_label'], [[now, n, cap]])
        path, last = os.path.join(OUT, 'official_expect.csv'), {}
        if os.path.exists(path):
            for r in csv.DictReader(open(path, encoding='utf-8')):
                last[r['slot']] = int(r['pct'])
        changed = [[now, s, p] for s, p in slots if last.get(s) != p]
        if changed:
            append('official_expect.csv', ['fetched_kst', 'slot', 'pct'], changed)
        print(now, n, cap, len(slots), 'slots')
    except Exception as e:                      # keep the job green; failures are written to errors.csv
        append('errors.csv', ['ts_kst', 'error'], [[now, repr(e)[:300]]])
        print('error', repr(e)[:300])
