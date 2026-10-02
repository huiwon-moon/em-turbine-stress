import re
import numpy as np
import pandas as pd

# ===============================================================
# 1~4호기 데이터 결합
# ===============================================================
all_dfs = []
case_offset = 0   # 호기별 caseID 누적 오프셋

for NUM in [1, 2, 3, 4]:
    df = pd.read_csv(fr'data\{NUM}호기.csv')
    df['time'] = pd.to_datetime(df['time'])

    # ── 컬럼명 앞 숫자 두 자리 제거 (예: '11abc' → 'abc')
    df.columns = [re.sub(r"^\d{2}", "", c) for c in df.columns]

    # ── caseID 생성 (시간 간격 기반)
    df['time_diff']  = df['time'].diff()
    threshold        = pd.Timedelta(minutes=5)
    df['is_new_case'] = (df['time_diff'] > threshold) | (df['time_diff'].isna())
    df['caseID']     = df['is_new_case'].cumsum() - 1
    df = df.drop(columns=['time_diff', 'is_new_case'])

    # ── 호기별 caseID에 오프셋 더해서 중복 방지
    df['caseID']  = df['caseID'] + case_offset
    case_offset   = df['caseID'].max() + 1   # 다음 호기 시작값

    df['NUM']     = NUM   # 어느 호기인지 추적용 (선택)
    df['time_idx'] = df.groupby('caseID').cumcount()

    df = (
            df.set_index('time')
            .groupby('caseID')
            .resample('1min')
            .mean(numeric_only=True)
            .reset_index()
        )
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv('data/full.csv')