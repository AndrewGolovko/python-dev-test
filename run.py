import argparse
import os

from file_processor import (FileProcessor, MaxFeatureAbsMeanDiff, MaxIndex,
                            Standardizer)

DATA_DIR = 'data'
TRAIN_FILE = 'train.tsv'
TEST_FILE = 'test.tsv'
OUTPUT_FILE = 'test_proc.tsv'
SEPARATOR = '\t'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '-t', '--train',
    default=os.path.join(DATA_DIR, TRAIN_FILE),
    help='Path to file with training data.',
)
parser.add_argument(
    '-i', '--input',
    default=os.path.join(DATA_DIR, TEST_FILE),
    help='Path to file with data to be processed.',
)
parser.add_argument(
    '-o', '--output',
    default=os.path.join(DATA_DIR, OUTPUT_FILE),
    help='Path to file, where processed data will be stored.',
)
parser.add_argument(
    '-c', '--chunk_size',
    default=128,
    help='Rows to process at a time.',
)
args = parser.parse_args()


if __name__ == '__main__':
    train_path = args.train
    test_path = args.input
    output_path = args.output
    reader_params = {
        'chunksize': args.chunk_size,
        'sep': SEPARATOR,
    }

    transformers = (
        Standardizer,
        MaxIndex,
        MaxFeatureAbsMeanDiff,
    )
    processor = FileProcessor(transformers, reader_params=reader_params)
    processor.train(train_path)
    processor.process(test_path, output_path)
