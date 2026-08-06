"""Normalize features"""

import numpy as np


def normalize(features):

    features_normalized = np.copy(features).astype(float)

    # 計算均值
    features_mean = np.mean(features, 0)

    # 計算標準差
    features_deviation = np.std(features, 0)

    # 標準化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
