"""올림픽공원 공연 일정 수집 -> data/events_ksponco.json
올림픽공원 홈페이지 공연/행사안내가 쓰는 목록 API를 2023년 1월부터 실행한 달의 8개월 뒤까지 월별로 호출합니다.
(브라우저 개발자 도구에서 같은 요청을 보내도 됩니다)"""
import json, datetime as dt, requests
URL = 'https://www.ksponco.or.kr/olympicpark/eventInfo/eventInfoListAjax.es'
today = dt.date.today()
last = today.year * 12 + today.month - 1 + 8
months = [f'{m // 12}{m % 12 + 1:02d}' for m in range(2023 * 12, last + 1)]
events = {}
for ym in months:
    r = requests.post(URL, json={'nowYM': ym, 'plcCd': ''}, headers={'Content-Type': 'application/json;charset=UTF-8'}, timeout=30)
    r.raise_for_status()
    for part in r.json():
        for it in (part or []):
            events[it['evtSeq']] = it
json.dump(list(events.values()), open('./data/events_ksponco.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(len(events), 'events', months[0], '~', months[-1])
