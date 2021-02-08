from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, NoReturn, Sequence, Type

import numpy as np
import pandas as pd

EPSILON = 1e-10


def pd_add(x: pd.Series, y: pd.Series) -> pd.Series:
    """Adds two pd.Series."""
    return x.add(y, fill_value=0)


class Statistic(ABC):
    """Base class for all statistics. Encapsulates logic of a statistic
    calculation.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of a statistic."""

    @abstractmethod
    def update(self, df: pd.DataFrame) -> NoReturn:
        """Updates statistic value using given data batch.

        :param df: pd.DataFrame, data batch.
        """

    @property
    @abstractmethod
    def value(self) -> pd.Series:
        """Returns current value of a statistic."""


class Moment(Statistic):
    """Base class for raw moments."""
    name = 'moment'

    @property
    @abstractmethod
    def order(self) -> int:
        """Order of the moment."""

    def __init__(self):
        self._count = None
        self._moment = None

    def update(self, df: pd.DataFrame):
        """Updates statistic value using given data batch.

        :param df: pd.DataFrame, data batch.
        """
        # old values
        if self._count is None:
            self._count = pd.Series(dtype='float64')
        if self._moment is None:
            self._moment = pd.Series(dtype='float64')

        # chunk values
        count = df.count(numeric_only=True)
        moment = (df ** self.order).mean(skipna=True, numeric_only=True)

        total_sum = pd_add(self._moment * self._count, moment * count)
        total_count = pd_add(self._count, count)

        # updated values
        self._count = total_count
        self._moment = total_sum / total_count.clip(lower=EPSILON)

    @property
    def value(self) -> pd.Series:
        """Returns current value of a moment."""
        return deepcopy(self._moment)

    @property
    def count(self) -> pd.Series:
        """Returns current count of data samples."""
        return deepcopy(self._count)


class Mean(Moment):
    """Calculates mean."""
    name = 'mean'
    order = 1


class SecondMoment(Moment):
    """Calculates second raw moment."""
    name = 'second_moment'
    order = 2


class StandardDeviation(Statistic):
    """Calculates standard deviation (with 1 degree of freedom)."""
    name = 'std'
    ddof = 1

    def __init__(self):
        self.mean = Mean()
        self.second_moment = SecondMoment()

    def update(self, df: pd.DataFrame):
        """Updates statistic value using given data batch.

        :param df: pd.DataFrame, data batch.
        """
        self.mean.update(df)
        self.second_moment.update(df)

    @property
    def value(self) -> pd.Series:
        """Returns current value of standard deviation."""
        return self._std(
            count=self.mean.count,
            mean=self.mean.value,
            second_moment=self.second_moment.value,
        )

    @classmethod
    def _std(
            cls,
            count: pd.Series,
            mean: pd.Series,
            second_moment: pd.Series
    ) -> pd.Series:
        """Calculates standard deviation using count, mean and second moment.

        :param count: pd.Series, accumulated count.
        :param mean: pd.Series, accumulated mean.
        :param second_moment: pd.Series, accumulated second moment.
        :return: pd.Series, standard deviation.
        """
        coef = count / (count - cls.ddof).clip(lower=EPSILON)
        variation = coef * (second_moment - np.square(mean))
        return np.sqrt(variation)


STATISTICS: Dict[str, Type[Statistic]] = {
    Mean.name: Mean,
    SecondMoment.name: SecondMoment,
    StandardDeviation.name: StandardDeviation
}


class StatisticsCalculator:
    """Calculates specified statistics.

    :param statistics: sequence of str, names of statistics to calculate.
    """
    def __init__(self, statistics: Sequence[str]):
        self._statistics = {name: STATISTICS[name]() for name in statistics}

    def update(self, df: pd.DataFrame):
        """Updates statistics using given data batch.

        :param df: pd.DataFrame, data batch.
        """
        for stat in self._statistics.values():
            stat.update(df)

    @property
    def statistics(self) -> Dict[str, pd.Series]:
        """Returns current values of statistics.

        :return: dict of str -> pd.Series, keys are names of statistics and
        values are values of statistics.
        """
        return {name: stat.value for name, stat in self._statistics.items()}
