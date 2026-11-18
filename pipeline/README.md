# 공개 시연판 매일 갱신

이 폴더는 https://kjinwater1-crypto.github.io/olpark/ 의 예보를 매일 새벽 다시 만드는 데 필요한 부분만 담았습니다.
GitHub Actions(.github/workflows/daily.yml)가 매일 04시 무렵 `python pipeline/run_site.py`를 실행해
올림픽공원 홈페이지의 공연 일정을 새로 받고, 저장된 모델로 예보를 다시 만든 뒤 맨 위의 index.html을 바꿉니다.

- 지난 날짜는 그날까지 공개했던 예보를 그대로 둡니다.
- 사전 공개 예보 채점(live/SCORING.md)은 live 브랜치의 실시간 기록으로 함께 갱신해 화면 '모델과 데이터'에 보여 줍니다.
- 데이터에는 10분 단위와 시간대별 합계만 있고 차량 식별값은 없습니다.
- 데이터 수집과 정제, 학습, 검증을 포함한 전체 원천 소스와 설명은 경진대회에 제출한 olpark_source.zip에 있습니다.
