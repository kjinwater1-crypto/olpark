"""Inject app_data.json (+ grouped feature importance) into app/template.html -> app/olpark-forecast.html.
OLPARK_VIEW=public writes app/olpark-public.html instead: the visitor screens only, for the public homepage
(the operator screens stay in the full file for the internal network)."""
import json, os, pandas as pd
D = './data/'
A = './app/'
app = json.load(open(D + 'app_data.json', encoding='utf-8'))
imp = pd.read_csv(D + 'feature_importance.csv', index_col=0).iloc[:, 0]
def grp(f):
    if f in ('rec_hour', 'rec_dpk'): return '최근 4주 이용 수준'
    if f.startswith(('laud_', 'press_', 'rel_')) or f in ('n_venues', 'aud_big3'): return '공연 관객 규모와 시작 시각'
    if f in ('hour', 'dow', 'is_hol', 'is_off', 'next_off', 'prev_off'): return '시간대와 요일, 휴일'
    if f in ('month', 'doy_sin', 'doy_cos', 'school_vac'): return '계절과 방학'
    if f in ('avg_temp', 'max_temp', 'min_temp', 'rain_mm', 'rainy', 'heavy_rain', 'avg_cloud'): return '날씨'
    if f == 'post_sys': return '주차 시스템 교체 시점'
    return '기타'
g = imp.groupby(imp.index.map(grp)).sum().sort_values(ascending=False)
app['imp'] = [[k, round(float(v), 1)] for k, v in g.items()]
app['inc'] = json.load(open(D + 'event_increment.json', encoding='utf-8'))
app.pop('resp', None)
app['imp_top'] = [[k, round(float(v), 1)] for k, v in imp.head(10).items()]
VIEW = os.environ.get('OLPARK_VIEW', 'all'); app['view'] = VIEW
s = json.dumps(app, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
tpl = open(A + 'template.html', encoding='utf-8').read()
out = tpl.replace('__APP_DATA__', s)
# standalone document: charset, phone viewport and the hidden rule that the views rely on
if not out.lstrip().lower().startswith('<!doctype'):
    cut = out.index('</style>') + len('</style>')          # title, meta, font links and the main stylesheet go to <head>
    out = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
           '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n' + out[:cut] + '\n</head><body>' + out[cut:] + '</body></html>')
open(A + ('olpark-public.html' if VIEW == 'public' else 'olpark-forecast.html'), 'w', encoding='utf-8').write(out)
print('built', len(out) // 1024, 'KB', app['imp'])
