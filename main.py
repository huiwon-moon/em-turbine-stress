import os
import sys
import random
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import List
from lightgbm import LGBMRegressor

from src.features import create_target_deltas, derived_variable, add_features
from src.model import LGBMMultiOutputModel, LinearRegressionModel
from src.evaluation import prepare_true_values, performance_evaluate, save_performance_matrix
from src.visualization import save_startup_plots
from config.columns import get_target_col

def parse_args():
    parser = argparse.ArgumentParser(description="Turbine Stress Pipeline")
    
    # 데이터 경로 지정
    parser.add_argument('--train_path', type=str, default='data/processed/train.csv', help='Train CSV 파일 경로')
    parser.add_argument('--test_path', type=str, default='data/processed/test.csv', help='Test CSV 파일 경로')
    parser.add_argument('--model_dir', type=str, default='models', help='학습된 모델 저장 디렉토리')
    
    # 모델 파라미터 대상
    parser.add_argument('--turbine', type=str, default='RH_IN', choices=['HP', 'RH_IN', 'RH_OUT'], help='터빈 영역')
    parser.add_argument('--part', type=str, default='bore', choices=['surf', 'bore'], help='부위 (e.g., bore, surface)')
    
    # LightGBM 하이퍼파라미터
    parser.add_argument('--n_estimators', type=int, default=10000, help='최대 트리 개수')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='학습률')
    parser.add_argument('--num_leaves', type=int, default=31, 
                        help='트리 하나당 최대 잎(Leaf) 노드 개수 (복잡도 제어 및 과적합 방지)')
    parser.add_argument('--min_child_samples', type=int, default=40, 
                        help='한 잎(Leaf) 노드가 가져야 하는 최소 데이터 개수 (노이즈 학습 방지)')
    parser.add_argument('--subsample', type=float, default=0.8, 
                    help='개별 트리를 학습할 때 사용할 데이터 샘플링 비율 (0.0~1.0, 과적합 방지 및 학습 속도 향상)')
    parser.add_argument('--random_state', type=int, default=42, help='seed')

    # 모델 정보
    parser.add_argument('--model_name', type=str, default='lgbm', choices=['lgbm', 'linear'], help='모델 이름')
    parser.add_argument('--task_type', type=str, default='LA_multi', choices=['single', 'LA_multi'], help='TBN STRESS: 단일 모델 / TBN LA STRESS: 다중 예측 모델')
    parser.add_argument('--output_dir', type=str, default='results', help='성능 평가 저장 경로')
    
    args, _ = parser.parse_known_args()
    
    return args

def prepare_features_and_targets(
        df: pd.DataFrame,
        target_cols: List[str],
        ignore_cols: List[str],
):
    all_drop_targets = list(set(target_cols + ignore_cols))
    drop_cols = [col for col in all_drop_targets if col in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[[col for col in target_cols if col in df.columns]]

    return X, y

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    args = parse_args()
    set_seed(args.random_state)

    print("=" * 65)
    print(f"[Train Pipeline Start] Turbine: {args.turbine} | Part: {args.part}")
    print("=" * 65)

    # 1. 데이터 로드
    print("\n 1. Loading Pre-split CSV Datasets...")
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Train dataset not found at: {args.train_path}")

    train_df = pd.read_csv(args.train_path, index_col=False)

    # Target 컬럼 정의 (Multi-horizon)
    mapping = {
            'HP': '1',
            'RH_IN': '2',
            'RH_OUT': '3'
        }
    turbine_num = mapping.get(args.turbine)

    target_col = get_target_col(args.part, turbine_num)

    # TBN LA STRESS 모델인 multi-horizon의 경우만 delta로 학습
    if args.task_type == 'LA_multi':
        horizons = [1, 2, 3, 4, 5]
        train_df, target_cols_list = create_target_deltas(train_df, target_col, group_col='caseID', horizons=horizons)
    else:
        target_cols_list = [target_col]
    
    # Feature Engineering
    train_df, feature_cols = derived_variable(train_df, args.turbine, args.part, turbine_num)
    train_df = add_features(train_df, feature_cols, lag_steps=30, task_type=args.task_type)

    # X, y 분리
    if args.task_type == 'LA_multi':
        ignore_cols = ['time', 'caseID']
    else:
        ignore_cols = [target_col, 'time', 'caseID', 'NUM']

    X_train, y_train = prepare_features_and_targets(train_df, target_cols_list, ignore_cols)
    
    print(f"   - Number of Input Features: {X_train.shape[1]}")

    # 2. 모델 학습
    if args.task_type == 'LA_multi':
        print("\n 2. Training LightGBM Model...")
        model = LGBMMultiOutputModel(
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
            args.min_child_samples,
            args.subsample,
            args.random_state,
        )
    else:
        print("\n 2. Training Linear Regression...")
        model = LinearRegressionModel()

    model.fit(X_train, y_train)

    # 3. 모델 저장
    print("\n 3. Saving Trained Model")
    model_save_dir = os.path.join(args.model_dir, args.task_type)
    model.save(
        model_name=args.model_name,
        turbine=args.turbine,
        part=args.part,
        task_type=args.task_type,
        save_dir=model_save_dir
    )

    print("=" * 65)
    print(f"[Inference Pipeline Start] Turbine: {args.turbine} | Part: {args.part}")
    print("=" * 65)

    # 4. Test 데이터 전처리 및 파이프라인 적용
    print("\n 4. Loading & Preprocessing Test Dataset...")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test dataset not found at: {args.test_path}")

    test_df = pd.read_csv(args.test_path)
    if args.task_type == 'LA_multi':
        test_df, _ = create_target_deltas(test_df, target_col, group_col='caseID', horizons=horizons)

    # Feature Engineering
    test_df, feature_cols = derived_variable(test_df, args.turbine, args.part, turbine_num)
    test_df = add_features(test_df, feature_cols, lag_steps=30, task_type=args.task_type)

    # X, y 분리
    X_test, y_test = prepare_features_and_targets(test_df, target_cols_list, ignore_cols)
    
    print(f"   - Number of Input Features: {X_test.shape[1]}")

    # 5. 추론 실행
    print("\n 5. Running Inference...")
    model_path = os.path.join(args.model_dir, args.task_type, f'{args.turbine}_{args.part}_{args.model_name}_{args.task_type}.pkl')
    if args.task_type == 'LA_multi':
        loaded_model = LGBMMultiOutputModel.load(model_path)
        y_pred = loaded_model.predict(X_test, base_target_series=test_df[[target_col]].values)
    else:
        loaded_model = LinearRegressionModel.load(model_path)
        y_pred = loaded_model.predict(X_test)

    # 6. 성능 평가 및 결과 저장
    file_name = f'{args.task_type}_performance_matrix'

    print("\n 6. Evaluating Predictions...")
    if args.task_type == 'LA_multi':
        y_true = prepare_true_values(test_df, target_col, target_cols_list)
    else:
        y_true = prepare_true_values(test_df, target_col)

    summary_df = performance_evaluate(test_df, y_true, y_pred, model_name=args.model_name, target_cols_list=target_cols_list)
    save_performance_matrix(summary_df, file_name, args.output_dir)

    # 7. 시각화 및 예측 결과 저장
    print("\n 7. Saving Results & Generating Plot...")
    save_startup_plots(
        test_df,
        y_true,
        y_pred,
        args.task_type,
        args.model_name,
        args.turbine,
        args.part,
        )

    print("\n" + "=" * 65)
    print("[Inference Pipeline Finished Successfully]")
    print("=" * 65)

if __name__ == '__main__':
    main()