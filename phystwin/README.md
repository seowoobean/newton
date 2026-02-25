# phystwin 실행 가이드

## 권장 실행 순서

### 1) 빠른 시뮬레이션 확인
```bash
bash phystwin/run_sim.sh
```
- 역할: `final_data.pkl`을 바로 읽어 spring-mass 시뮬레이션만 확인합니다.
- 실행 대상: `phystwin/sim/spring_mass_from_pkl.py`

### 2) (선택) CMA로 초기 파라미터 탐색
```bash
bash phystwin/run_optimize_cma.sh
```
- 역할: 스프링/입자 관련 초기 파라미터를 탐색해 `optimal_params.pkl`을 만듭니다.
- 실행 대상: `phystwin/optimize/optimize_cma.py`

### 3) 학습(역물리)으로 스프링 파라미터 추정
```bash
bash phystwin/run_train.sh
```
- 역할: `spring_stiffness`, `spring_damping` 등을 학습해 `best_params.pkl`을 만듭니다.
- 실행 대상: `phystwin/train/train_warp.py`

### 4) 인터랙티브 실행용 모델 PKL 내보내기
```bash
bash phystwin/run_export.sh
```
- 역할: `final_data.pkl` + config + `optimal_params.pkl` + `best_params.pkl`를 합쳐 `model.pkl`을 생성합니다.
- 실행 대상: `phystwin/sim/export_model.py`

### 5) 로봇-클로스 인터랙티브 실행
```bash
bash phystwin/run_interactive.sh
```
- 역할: URDF 로봇 + spring-mass를 함께 띄워 접촉/드래그를 확인합니다.
- 실행 대상: `phystwin/sim/spring_mass_interactive.py`

## 스크립트별 역할 요약

- `phystwin/run_sim.sh`: 단일 spring-mass 시뮬레이션 확인
- `phystwin/run_optimize_cma.sh`: 초기 물리 파라미터(CMA) 탐색
- `phystwin/run_train.sh`: 학습으로 스프링 파라미터 추정
- `phystwin/run_export.sh`: interactive용 `model.pkl` 생성
- `phystwin/run_interactive.sh`: 로봇 + spring-mass 인터랙티브 실행

## 케이스 변경

각 `run_*.sh`의 `CASE_NAME`을 같은 값으로 맞춰서 사용하세요.
예: `double_lift_cloth_1`
