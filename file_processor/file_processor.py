import os
from typing import Iterable, Sequence, Type

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from file_processor.statistics import StatisticsCalculator
from file_processor.transformers import Transformer


class FileReader:
    """Reads data from text file chunk by chunk.

    :param path: str, path to file to read data from.
    :param pandas_kwargs: keywords arguments for Pandas method `read_csv()`.
    """
    def __init__(self, path, **pandas_kwargs):
        self.path = path
        self.pandas_kwargs = {'chunksize': 256, **pandas_kwargs}
        self.n_chunks = sum(1 for _ in self.read())

    def read(self) -> Iterable:
        """Reads and returns data from file."""
        return pd.read_csv(self.path, **self.pandas_kwargs)

    def chunks(self, desc: str = '') -> Iterable[pd.DataFrame]:
        """Reads data from file and returns it in unrolled form with each
        column corresponding a single feature entry.

        :param desc: str, prefix for tqdm progressbar (default '').
        :return: iterable of pd.DataFrame, iterator for data chunks.
        """
        return tqdm(
            map(self.pivot, self.read()),
            total=self.n_chunks,
            desc=desc
        )

    @staticmethod
    def pivot(df: pd.DataFrame) -> pd.DataFrame:
        """Converts dataframe into unrolled form with each column
        corresponding a single feature entry.

        :param df: pd.DataFrame, raw data.
        Must have columns 'id_job' and 'features'. Column 'features' contains
        comma separated list of values. First value is a feature name and
        others are entries of this feature vector.
        :return: pd.DataFrame, data in unrolled form.
        """
        df = df.copy()
        df['features'] = df['features'].str.split(',')
        df['name'] = df['features'].apply(lambda x: x[0])
        df['features'] = df['features'].apply(lambda x: x[1:])

        length = df['features'].apply(len)
        if length.min() != length.max():
            raise ValueError(f'Feature vectors have different lengths '
                             f'(from {length.min()} to {length.max()}).')

        exploded = df.explode('features')
        indices = np.tile(np.arange(length.min()), df.shape[0]).astype(str)
        exploded['name'] = 'feature_' + exploded['name'] + '_' + indices

        return pd.pivot(
            data=exploded,
            index='id_job',
            columns='name',
            values='features'
        ).astype('float64')


class FileProcessor:
    """Applies specified transformation to the data from text file.

    :param transformers: sequence of Transformer types, transformations to
    be applied to the data.
    :param reader_params: dict, arguments to be passed to
    `pandas.read_csv` method while reading from file.
    """
    output_options = {
        'sep': '\t',
        'mode': 'a',
        'index': False
    }

    def __init__(
            self,
            transformers: Sequence[Type[Transformer]],
            reader_params: dict
    ):
        self.transformer_types = transformers
        self.transformers = None
        self.statistics = StatisticsCalculator([
            stat for t in self.transformer_types
            for stat in t.statistics
        ])
        self.reader_params = reader_params

    def train(self, train_path: str):
        """Learns statistics using data from text file.

        :param train_path: str, path to file with data.
        """
        reader = FileReader(train_path, **self.reader_params)
        for pivot_df in reader.chunks(desc='training'):
            self.statistics.update(pivot_df)

        self.transformers = [
            cls(**self.statistics.statistics)
            for cls in self.transformer_types
        ]

    def process(self, input_path: str, output_path: str):
        """Reads file, applies transformations to the data from file and
        saves results to another file.

        :param input_path: str, path to file to be processed.
        :param output_path: str, path to file for saving results.
        """
        if os.path.exists(output_path):
            os.remove(output_path)

        header = True
        reader = FileReader(input_path, **self.reader_params)
        for chunk in reader.chunks(desc='processing'):
            pd.concat(
                [t.transform(chunk) for t in self.transformers],
                axis=1
            ).reset_index(
            ).to_csv(output_path, header=header, **self.output_options)

            header = False
