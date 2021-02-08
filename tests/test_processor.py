import os
from tempfile import TemporaryDirectory
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pytest

from file_processor import (FileProcessor, MaxFeatureAbsMeanDiff, MaxIndex,
                            Standardizer)


def generate_data(
        n_rows: int,
        feature: Union[str, int],
        feature_size: int,
        seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generates random features.

    :param n_rows: int, number of rows.
    :param feature: str or int, feature name.
    :param feature_size: int, size of feature vector.
    :param seed: int, random seed.
    :return df: pd.DataFrame, generated data.
    :return values: np.ndarray, 2D array of feature values.
    """
    rand = np.random.RandomState(seed)

    feature_name = np.array([feature]).repeat(n_rows)
    shape = (n_rows, feature_size)
    values = rand.randint(-3000, 1000, size=shape)

    features = pd.concat(
        [pd.Series(feature_name), pd.DataFrame(values)],
        axis=1
    ).astype(str).agg(','.join, axis=1)
    features.index.name = 'id_job'
    df = features.reset_index(name='features')

    return df, values


@pytest.mark.parametrize(
    'feature_size, chunksize',
    [
        (1, 1),
        (1, 16),
        (16, 1),
        (16, 16),
        (16, 49),
        (16, 50),
        (16, 51),
        (16, 128),
        (256, 32),
        (256, 50),
        (256, 64),
        (256, 256),
    ]
)
def test_processor(feature_size, chunksize):
    """Tests FileProcessor.

    :param feature_size: int, size of feature vector.
    :param chunksize: int, FileProcessor chunk size.
    """
    sep = '\t'
    n_rows = 50
    feature = 3

    with TemporaryDirectory() as dir:
        input_path = os.path.join(dir, 'data.tsv')
        output_path = os.path.join(dir, 'data_proc.tsv')

        data, feature_values = generate_data(
            n_rows=n_rows,
            feature=feature,
            feature_size=feature_size,
            seed=42
        )

        data.to_csv(input_path, sep=sep, index=False)

        reader_params = {
            'chunksize': chunksize,
            'sep': sep,
        }

        transformers = (
            Standardizer,
            MaxIndex,
            MaxFeatureAbsMeanDiff,
        )
        processor = FileProcessor(transformers, reader_params=reader_params)
        processor.train(input_path)
        processor.process(input_path, output_path)

        processed = pd.read_csv(output_path, sep=sep)

    # check feature_{i}_stand_{index}
    expected_stand = (feature_values - feature_values.mean(axis=0)
                      ) / feature_values.std(axis=0, ddof=1)
    stand = processed.filter(regex=f'feature_{feature}_stand_[0-9]+')
    assert np.allclose(expected_stand, stand)
    assert np.allclose(stand.mean(axis=0), 0)
    assert np.allclose(stand.std(axis=0, ddof=1), 1)

    # check max_feature_{i}_index
    expected_max = feature_values.max(axis=1)
    max_index = processed[f'max_feature_{feature}_index'].values
    max_mask = (
            max_index.reshape((-1, 1)) ==
            np.arange(feature_values.shape[1]).reshape((1, -1))
    )
    fact_max = feature_values[max_mask]
    assert np.allclose(expected_max, fact_max)

    # check max_feature_{i}_abs_mean_diff
    expected_max_mean = np.broadcast_to(feature_values.mean(axis=0),
                                        shape=max_mask.shape
                                        )[max_mask]
    expected_abs_mean_diff = np.abs(expected_max - expected_max_mean)
    abs_mean_diff = processed[f'max_feature_{feature}_abs_mean_diff']
    assert np.allclose(expected_abs_mean_diff, abs_mean_diff)
