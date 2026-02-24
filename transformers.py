import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):

    def __init__(self, cols, lower_q=0.01, upper_q=0.99):
        self.cols = cols
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.clip_limits_ = {}

    def fit(self, X, y=None):
        # Đảm bảo X là DataFrame để truy cập theo tên cột
        X_ = X.copy()
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_)

        self.clip_limits_ = {}
        for col in self.cols:
            if col in X_.columns:
                lower = X_[col].quantile(self.lower_q)
                upper = X_[col].quantile(self.upper_q)
                self.clip_limits_[col] = (lower, upper)
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_)

        # Sử dụng các ngưỡng đã được lưu trong self.clip_limits_ khi fit/load
        for col, (lo, hi) in self.clip_limits_.items():
            if col in X_.columns:
                X_[col] = X_[col].clip(lo, hi)
        return X_

    def get_feature_names_out(self, input_features=None):
        return input_features