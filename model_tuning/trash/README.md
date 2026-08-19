# Archived model-tuning code

이 폴더는 현재 `real_car_v2` numbered pipeline에서 사용하지 않는 legacy 코드를 보존한다.
삭제된 코드는 없으며 필요하면 git history와 함께 복원할 수 있다.

- `legacy_top_level/`: 초기 bag 분석, 구형 모델 비교, 수동 replay 및 이전 plotting entry points
- `legacy_utils/`: `model_tuning_utils`의 구형 generic extraction/training framework
- `legacy_real_car/`: canonical pipeline과 중복된 이전 runner/trainer

이 코드는 현재 파일 위치에서 실행 가능하다고 보장하지 않는다. 재사용하려면 먼저
`real_car_v2`의 data/sign/latency contract에 맞는지 검토한 뒤 명시적으로 복원한다.
