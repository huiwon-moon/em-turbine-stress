# Turbine Stress Forecasting & Estimation Pipeline

터빈 스트레스 단일 추정(**TBN STRESS**) 및 다단계 미래 예측(**TBN LA STRESS**)을 위한 시계열 ML 파이프라인 시스템입니다.

---

## Key Features

- **다중 타겟/작업 지원**:
  - `TBN STRESS`: 현재 시점 단일 추정 (Single-point Estimation)
  - `TBN LA STRESS`: 미래 $t+1 \sim t+5$ 스텝 Multi-horizon 예측
- **Feature Engineering 파이프라인**:
  - `caseID` 단위 시계열 그룹핑 기반 Lag, Diff, Rolling (Std, Max, Min, Range) 자동 생성
- **통합 CLI & Orchestration**:
  - `main.py`를 통합 제어 타워로 사용하여 `train`, `inference` 모드 일괄 제어

---

## Directory Structure

```text
├── data/
│   ├── raw/                # (비공개) 원천 시계열 CSV 데이터
│   └── processed/          # (비공개) Pre-split 전처리 데이터 (train.csv, test.csv)
├── models/
│   └── LA_multi/           # (비공개) 학습 완료된 TBN LA STRESS LightGBM 모델 (.pkl)
│   └── single/             # (비공개) 학습 완료된 TBN STRESS LinearRegressor 모델 (.pkl)
├── config/
│   └── columns.py          # (비공개) 타겟 태그 정보
├── docs/                   # 기술 설명서
├── fig/
│   └── LA_multi/           # (비공개) TBN LA STRESS LightGBM 평가 시각화
│   └── single/             # (비공개) TBN STRESS LinearRegressor 평가 시각화
├── results/                # 추론 결과 CSV
├── src/
│   └── data_concat.py      # 원천 시계열 CSV 데이터 결합
│   └── data_split.py       # 훈련/테스트 데이터 분할
│   └── evaluation.py       # 성능 평가 코드
│   └── features.py         # (비공개) feature engineering
│   └── model.py            # 모델 구조
│   └── visualization.py    # 시각화 코드
├── main.py                 # 파이프라인 실행 진입점 (Entry Point)
└── requirements.txt        # 의존성 라이브러리 목록
```
---

## Usage
### 1. 가상환경 생성 및 라이브러리 설치
```bash
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 파이프라인 실행
```bash
python main.py --turbine HP --part surf --model_name lgbm --task_type LA_multi
```

---

## Validation & Saving
- **추론 메트릭**: `results/` 폴더 내에 Horizon별로 MAE, RMSE 메트릭이 포함된 CSV 파일 저장
- **시각화 결과**: 실제 TBN STRESS vs 모델 예측값 시계열 그래프 생성 (.png)

---

## CLI Arguments

| 옵션 (Argument) | 타입 (Type) | 기본값 (Default) | 선택 가능 값 (Choices) | 설명 (Description) |
| :--- | :--- | :--- | :--- | :--- |
| `--turbine` | `str` | `RH_IN` | `HP`, `RH_IN`, `RH_OUT` | 대상 터빈 영역 |
| `--part` | `str` | `bore` | `bore`, `surf` | 대상 터빈 부위 |
| `--n_estimators` | `int` | `10000` | 양의 정수 | LightGBM 트리의 최대 개수 |
| `--learning_rate` | `float` | `0.01` | 양의 실수 | LightGBM 모델 학습률 |
| `--random_state` | `int` | `42` | 정수 | 재현성을 위한 난수 고정 시드값 |
| `--model_name` | `str` | `lgbm` | `lgbm`, `linear` | 모델 종류 |
| `--task_type` | `str` | `LA_multi` | `single`, `LA_multi` | TBN STRESS 또는 TBN LA STRESS |
