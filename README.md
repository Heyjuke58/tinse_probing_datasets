# tinse_probing_datasets
Script(s) to generate datasets for probing tasks of BERT (project TINSE)



## Setup

1. ```conda create -n tinse python=3.8```

python -m spacy download en_core_web_sm
TODO


***

## Usage

```conda activate tinse```

### Creating datasets

Example script run, which creates datasets (in /datasets) for all tasks for one specified source dataset (e.g. msmarco)

```python src/msmarco_dataset_creation.py -s 10000 -sq 5 -src msmarco```

####Options
| Option      | Description | Default  |
| ----------- | ----------- | ----------- |
| -s, --size      | Size of the generated dataset(s) | 10000|
| -sq, --samples_per_query   | Determines the maximumn number of passage samples with the same query in the generated dataset         |  5 |
| -src, --source   | Source Dataset  | msmarco |

.. to be continued

### Get dataset splits

TODO

***

## Directory Contents

### assets
source datasets (msmarco, trec, etc.)

### datasets
generated datasets in json format

***

## Other useful commands

#### Run detached script on server:
```tmux```
run script

To quit tmux session:
Ctrl+b then d

To reattach to tmux session:
```tmux attach```

#### Get large files from google drive:

```gdown https://drive.google.com/uc?id=<file_id>```

Problem:
Kein PUT Access um neuen Index zu erstellen