import os
import joblib
import numpy as np
import pandas as pd
from typing import List
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import lightgbm as lgbm


class LGBMMultiOutputModel:
    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            num_leaves: int,
            min_child_samples: int,
            subsample: float,
            random_state: int
    ):
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': -1,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'subsample_freq': 0,
            'random_state': random_state,
            'n_jobs': -1
        }

        self.base_model = lgbm.LGBMRegressor(**self.params)
        self.model = MultiOutputRegressor(self.base_model)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """모델 학습"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame, base_target_series: pd.Series = None):
        """예측 수행"""
        if not self.is_fitted:
            raise RuntimeError("모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요.")

        delta_pred = self.model.predict(X_test)

        if base_target_series is not None:
            base_val = base_target_series.reshape(-1, 1)
            actual_pred = delta_pred + base_val
            return actual_pred

    def save(
        self,
        model_name: str,
        turbine: str,
        part: str,
        task_type: str,
        save_dir: str = 'models'
    ):
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{turbine}_{part}_{model_name}_{task_type}.pkl')
        joblib.dump(self, model_path)
        print(f"Model saved to: {model_path}")
        return model_path

    @classmethod
    def load(cls, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        loaded_instance = joblib.load(model_path)
        print(f"Model successfully loaded from: {model_path}")
        return loaded_instance

class LinearRegressionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, X_trian: pd.DataFrame, y_train: pd.DataFrame):
        X_train_scaled = self.scaler.fit_transform(X_trian)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요.")

        X_test_scaled = self.scaler.transform(X_test)
        pred = self.model.predict(X_test_scaled)

        return pred

    def save(
            self,
            model_name: str,
            turbine: str,
            part: str,
            task_type: str,
            save_dir: str = 'models'
        ):
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{turbine}_{part}_{model_name}_{task_type}.pkl')
            joblib.dump(self, model_path)
            print(f"Model saved to: {model_path}")
            return model_path
    
    @classmethod
    def load(cls, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        loaded_instance = joblib.load(model_path)
        print(f"Model successfully loaded from: {model_path}")
        return loaded_instance
        