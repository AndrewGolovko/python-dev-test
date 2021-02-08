import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


def get_column_index(name: str):
    """Returns index of a feature entry given its name.

    :param name: str, name of column.
    :return: int, index.
    """
    return int(re.findall('[0-9]+$', name)[0])


class Transformer(ABC):
    statistics = ()

    def __init__(self, **statistics):
        """Base class for all transformers. Transformer applies a data
        transformation using provided statistics of the data.

        :param statistics: keyword arguments of statistics required by
        transformation. Required statistics are defined by `statistics`
        attribute.
        """
        for stat in self.statistics:
            if stat not in statistics:
                raise ValueError(
                    f'{self.__class__.__name__} requires '
                    f'{self.statistics} to be initialized, '
                    f'got {tuple(statistics)} instead.'
                )

        for stat, value in statistics.items():
            if stat in self.statistics:
                setattr(self, stat, value)

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies transformation on the provided data

        :param df: pd.DataFrame, data to apply transformation on. Each column
        must correspond to a single entry of a feature vector.
        :return: pd.DataFrame, transformed data.
        """

    @staticmethod
    def get_features(df: pd.DataFrame) -> List[str]:
        """Returns list with names of features in DataFrame

        :param df: pd.DataFrame, data to apply transformation on. Each column
        must correspond to a single entry of a feature vector.
        :return: list of str, names of features.
        """
        return list(set([col.split('_')[1] for col in df.columns]))


class Standardizer(Transformer):
    """Applies standardization.

    :param mean: pd.Series, mean values of features.
    :param std: pd.Series, standard deviations of features.
    """
    statistics = ('mean', 'std')

    def transform(self, df) -> pd.DataFrame:
        """Applies standardization.

        :param df: pd.DataFrame, data to apply transformation on. Each column
        must correspond to a single entry of a feature vector.
        :return: pd.DataFrame, transformed data.
        """
        df_norm = (df - self.mean) / self.std
        df_norm.columns = [self.rename_column(col) for col in df_norm.columns]
        return df_norm[sorted(df_norm.columns, key=get_column_index)]

    @staticmethod
    def rename_column(column: str):
        """Returns name of standardized column.

        :param column: str, raw column name.
        :return: str, name of standardized column.
        """
        feature, name, index = column.split('_')
        return f'{feature}_{name}_stand_{int(index)}'


class MaxIndex(Transformer):
    """Finds index of maximum feature entry."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finds index of maximum feature entry.

        :param df: pd.DataFrame, data to apply transformation on. Each column
        must correspond to a single entry of a feature vector.
        :return: pd.DataFrame, indices of maximum feature entries.
        """
        return pd.concat([
            self.max_index(df, feature)
            for feature in self.get_features(df)
        ], axis=1)

    @staticmethod
    def max_index(df: pd.DataFrame, feature: str) -> pd.Series:
        """Finds index of maximum entry of a given feature.

        :param df: pd.DataFrame, data to apply transformation on. Each column
        must correspond to a single entry of a feature vector.
        :param feature: str, name of feature to find maximum.
        :return: pd.Series, indices of maximum feature entries.
        """
        return df.filter(regex=f'feature_{feature}_[0-9]+')\
            .idxmax(axis=1)\
            .apply(get_column_index)\
            .rename(f'max_feature_{feature}_index')


class MaxFeatureAbsMeanDiff(Transformer):
    """Calculates absolute difference between maximum entry of a feature and
    its mean.

    :param mean: pd.Series, mean values of features.
    """
    statistics = ('mean',)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates absolute difference between maximum entry of a feature and
        its mean.

        :param df: pd.DataFrame, data to apply transformation on. Each column
        must correspond to a single entry of a feature vector.
        :return: pd.DataFrame, absolute difference between maximum entry of a
        feature and its mean.
        """
        df_abs_mean_diff = pd.DataFrame()
        df_max_index = MaxIndex().transform(df)
        for feature in self.get_features(df):
            # get data for current feature
            feature_regex = f'feature_{feature}_[0-9]+'
            mean = self.mean.filter(regex=feature_regex)
            df_ = df.filter(regex=feature_regex)

            # sort features by indices
            mean = mean[sorted(mean.index, key=get_column_index)].values
            df_ = df_[sorted(df_.columns, key=get_column_index)]

            # get mask for max elements
            argmax = df_max_index[f'max_feature_{feature}_index'].values
            mask = (
                    argmax.reshape((-1, 1)) ==
                    np.arange(mean.size).reshape((1, -1))
            )

            # get max elements
            max_values = df_.values[mask]
            # get mean values for max features
            max_mean = np.broadcast_to(mean.reshape((1, -1)), mask.shape)[mask]
            # get absolute difference
            abs_diff = np.abs(max_values - max_mean)
            column_name = f'max_feature_{feature}_abs_mean_diff'
            df_abs_mean_diff[column_name] = pd.Series(abs_diff, index=df.index)
        return df_abs_mean_diff
