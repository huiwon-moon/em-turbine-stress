import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Union

# 폰트 및 유니코드 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def get_turbine_name(turbine: str):
    mapping = {
        'RH_IN': 'RH TBN INLET',
        'RH_OUT': 'RH RBN OUTLET'
    }
    return mapping.get(turbine, 'HP TBN')

def get_last_case_per_num(
        df: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray
):

    df = df.reset_index(drop=True)
    y = np.array(y)
    y_pred = np.array(y_pred)

    masks = []
    for num_id, group in df.groupby('NUM'):
        max_case = group['caseID'].max()
        mask = (df['NUM'] == num_id) & (df['caseID'] == max_case)
        masks.append(mask)

    combined_mask = pd.concat(masks, axis=0).groupby(level=0).any()
    mask_vals = combined_mask.values

    return(
        df[combined_mask].reset_index(drop=True),
        y[mask_vals],
        y_pred[mask_vals],
    )

def plot_startup_by_num_base(
        test_df: pd.DataFrame,
        y_test: Union[np.ndarray, list],
        y_test_pred: Union[np.ndarray, list],
        plot_all_horizons: bool,
        task_type: str,
        file_name: str,
        max_chunks: int = 100,
        ylabel: str = 'tbn stress',
        save_dir: str = 'fig',
        folder_type: str = 'full',
        horizon_idx: int = -1,
):
    y_test = np.array(y_test)
    y_test_pred = np.array(y_test_pred)
    test_df = test_df.reset_index(drop=True)

    is_1d = (y_test.ndim == 1)
    if is_1d:
        y_test_2d = y_test.reshape(-1, 1)
        y_test_pred_2d = y_test_pred.reshape(-1, 1)
        plot_all_horizons = False
        target_h_idx = 0
    else:
        y_test_2d = y_test
        y_test_pred_2d = y_test_pred
        target_h_idx = horizon_idx if 0 <= horizon_idx < y_test_2d.shape[1] else (y_test_2d.shape[1] - 1)

    if not pd.api.types.is_datetime64_any_dtype(test_df['time']):
        test_df['time'] = pd.to_datetime(test_df['time'])

    figs = {}

    for num_id, num_group in test_df.groupby('NUM'):
        num_id = int(num_id)

        save_path = os.path.join(save_dir, folder_type)
        os.makedirs(save_path, exist_ok=True)

        case_ids = num_group['caseID'].unique()[:max_chunks]
        nrows = len(case_ids)

        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(15, 5*nrows))
        if nrows == 1:
            axes = [axes]

        for ax, cid in zip(axes, case_ids):
            mask = test_df['caseID'] == cid
            chunk_df = test_df[mask]
            
            chunk_true = y_test_2d[mask.values]
            chunk_pred = y_test_pred_2d[mask.values]

            time_series = chunk_df['time']

            if task_type == 'LA_multi':
                time_series = time_series + pd.Timedelta(minutes=5)

            time_start = time_series.iloc[0]
            time_end = time_series.iloc[-1]

            actual_label = 'Actual(5m)' if plot_all_horizons else 'Actual'
            ax.plot(time_series, chunk_true[:, target_h_idx], 'b-', linewidth=2, label=actual_label)

            if plot_all_horizons and not is_1d:
                colors = ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#cc0000']
                for h in range(chunk_pred.shape[1]):
                    ax.plot(
                        time_series, chunk_pred[:, h],
                        color=colors[h], linestyle='--', linewidth=1.5,
                        label=f'Pred H{h+1}'
                    )
            else:
                ax.plot(time_series, chunk_pred[:, target_h_idx], 'r--', linewidth=2, label='Predicted')

            # Subplot 스타일 설정
            ax.set_title(
                f"({num_id}호기) {time_start.strftime('%Y-%m-%d %H:%M')} ~ {time_end.strftime('%H:%M')}",
                fontsize=12
            )
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        plt.tight_layout()

        fig_path = os.path.join(save_path, f'{num_id}호기_{file_name}')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close(fig)

        print(f"저장 완료: {fig_path}")

    return figs

def save_startup_plots(
        test_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str,
        model_name: str,
        turbine: str,
        part: str,
        base_save_dir: str = 'fig',
):
    turbine_name = get_turbine_name(turbine)
    ylabel = f'{turbine_name} {part.upper()} STRESS'
    
    prefix = f'{turbine}_{part}_{model_name}_{task_type}'

    # 1. Full 데이터 5분 예측 / 단일 예측 플롯
    plot_startup_by_num_base(
        test_df, y_true, y_pred,
        plot_all_horizons=False,
        task_type=task_type,
        ylabel=ylabel,
        save_dir=base_save_dir,
        folder_type=f'{task_type}/full',
        file_name=f'{prefix}'
    )

    # 2. Recent 데이터 플롯
    last_df, last_y, last_pred = get_last_case_per_num(test_df, y_true, y_pred)
    plot_startup_by_num_base(
        last_df, last_y, last_pred,
        plot_all_horizons=False,
        task_type=task_type,
        ylabel=ylabel,
        save_dir=base_save_dir,
        folder_type=f'{task_type}/recent',
        file_name=f'{prefix}'
    )

    # Full All Plot
    if task_type == 'LA_multi' and np.ndim(y_pred) > 1:
        plot_startup_by_num_base(
            test_df, y_true, y_pred,
            plot_all_horizons=True,
            task_type=task_type,
            ylabel=ylabel,
            save_dir=base_save_dir,
            folder_type=f'{task_type}/full',
            file_name=f'{prefix}_all_horizons'
        )
    
    print(f"All plots saved to base directory: {base_save_dir}")


