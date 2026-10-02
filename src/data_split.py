import os
import pandas as pd
from typing import List

def split_and_save_time_series_data(
        df: pd.DataFrame,
        output_dir: str = 'data/processed',
        train_years: List[int] = [2023, 2024],
        test_years: List[int] = [2025],
        time_col: str = 'time'
        ):

    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    train_df = df[df['time'].dt.year.isin(train_years)]
    test_df = df[df['time'].dt.year.isin(test_years)]

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Data split completed & saved to '{output_dir}/'")
    print(f"   - Train ({train_years}): {len(train_df):,} rows -> {train_path}")
    print(f"   - Test  ({test_years}): {len(test_df):,} rows -> {test_path}")

if __name__ == '__main__':
    # 예시 실행
    raw_df = pd.read_csv('data/raw/full.csv', index_col=0)
    split_and_save_time_series_data(raw_df)

    