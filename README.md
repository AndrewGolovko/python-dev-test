### Requirements

Project was tested under `python==3.7`.

Install requirements needed to run the script using command:
```
make requirements
```


### Run

Run script with default parameters:
```
python run.py
```

Use `--help` option to get information about parameters:

```
$ python run.py --help
usage: run.py [-h] [-t TRAIN] [-i INPUT] [-o OUTPUT] [-c CHUNK_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Path to file with training data. (default:
                        data/train.tsv)
  -i INPUT, --input INPUT
                        Path to file with data to be processed. (default:
                        data/test.tsv)
  -o OUTPUT, --output OUTPUT
                        Path to file, where processed data will be stored.
                        (default: data/test_proc.tsv)
  -c CHUNK_SIZE, --chunk_size CHUNK_SIZE
                        Rows to process at a time. (default: 128)
```
