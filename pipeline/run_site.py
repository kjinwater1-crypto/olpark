"""매일 새벽 GitHub Actions가 실행합니다: 올림픽공원 홈페이지에서 공연 일정을 새로 받아 예보를 다시 만들고,
저장소 맨 위의 index.html(공개 시연판)을 만듭니다. 지난 날짜는 그날까지 공개했던 예보를 그대로 둡니다.
사전 공개 예보의 채점(live/SCORING.md)도 live 브랜치의 실시간 기록으로 함께 갱신합니다.
원천 소스 전체와 설명은 제출한 olpark_source.zip에 있고, 이 폴더는 공개 시연판 갱신에 필요한 부분만 담았습니다.
차량 식별값은 들어 있지 않습니다(10분 단위와 시간대별 합계만)."""
import datetime as dt, os, shutil, subprocess, sys
P = os.path.dirname(os.path.abspath(__file__))
os.chdir(P)
today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
start = dt.date(2026, 11, 6)
end = max(dt.date(2027, 4, 30), today + dt.timedelta(days=150))


def run(script, *args, env=None, must=True):
    print(f'=== {script} {" ".join(args)}', flush=True)
    r = subprocess.run([sys.executable, os.path.join('scripts', script), *args], env=dict(os.environ, **(env or {})))
    if r.returncode and must:
        sys.exit(f'{script} failed ({r.returncode}); the published page is left as it was')
    return r.returncode == 0


run('00_fetch_events.py', must=False)          # if the homepage does not answer, the stored show list is used
run('05_parse_schedule.py')
run('07_event_features.py')
os.makedirs('data/published', exist_ok=True)
shutil.copy('data/app_data.json', 'data/published/prev_app_data.json')
run('23_live_prereg.py', must=False)
run('20_daily_forecast.py', start.isoformat(), str((end - start).days + 1),
    env={'OLPARK_TODAY': today.isoformat(), 'OLPARK_PUBLISHED': 'data/published/prev_app_data.json'})
shutil.copy('app/olpark-forecast.html', os.path.join('..', 'index.html'))
print('done', today, start, end)
