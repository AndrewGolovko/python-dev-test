"""Script for checking out the performance of FileProcessor on large files.
"""

import os
from tempfile import TemporaryDirectory

from tqdm.auto import tqdm

from file_processor import (FileProcessor, MaxFeatureAbsMeanDiff, MaxIndex,
                            Standardizer)
from tests.test_processor import generate_data

if __name__ == '__main__':
    sep = '\t'
    feature = 3

    with TemporaryDirectory() as dir:
        input_path = os.path.join(dir, 'data.tsv')
        output_path = os.path.join(dir, 'data_proc.tsv')

        header = True
        for seed in tqdm(range(100), desc='generating data'):
            data, _ = generate_data(
                n_rows=100_000,
                feature=feature,
                feature_size=512,
                seed=seed
            )
            data.to_csv(
                input_path,
                sep=sep,
                index=False,
                mode='a',
                header=header
            )
            header = False

        reader_params = {
            'chunksize': 10_000,
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
