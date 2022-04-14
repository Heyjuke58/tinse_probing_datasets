# tinse_probing_datasets
Script(s) to generate datasets for probing tasks of BERT (project TINSE)



## Setup

Make sure you have access to the docker daemon.

```sh
conda create -n tinse python=3.8
conda activate tinse
pip install -r requirements.txt
```

To install neuralcoref from source:
```sh
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
```

Download needed spacy pipelines:

```sh
python -m spacy download en_core_web_sm
```

***

## Usage
In other Terminal:
```sh
docker-compose up
```
```sh
docker run -p 12375:9200 -p 12376:9300 -e "discovery.type=single-node" --detach --name es -v esdata1:/usr/share/elasticsearch/data:rw  docker.elastic.co/elasticsearch/elasticsearch:7.16.2
```

```sh
conda activate tinse
python dataset_creation.py -t=<tasks> -s=<size> -sq=<samples_per_query> ...
```


### Creating datasets

Example script run, which creates datasets (in /datasets) for all tasks for one specified source dataset (e.g. msmarco)

Naming of Datasets: ``{source}_{task}_{size}_{samples_per_query}_{timestamp}.json``

```python src/msmarco_dataset_creation.py -s 10000 -sq 5 -src msmarco```

#### Options
| Option      | Description | Default  |
| ----------- | ----------- | ----------- |
| -s, --size      | Size of the generated dataset(s) | 10000|
| -sq, --samples_per_query   | Determines the maximumn number of passage samples with the same query in the generated dataset         |  5 |
| -src, --source   | Source Dataset  | msmarco |
| -t, --tasks   | Tasks to generate datasets for. Possible tasks are: ['bm25', 'semsim', 'ner']. Should be comma seperated  | bm25,semsim,ner |
| -ph, --port_http   | Http Port for elasticsearch container  | 12375 |
| -pt, --port_tcp   | TCP Port for elasticsearch container  | 12376 |

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

1. ```tmux```

2. run script

3. To quit tmux session: <kbd>Ctrl</kbd> + <kbd>b</kbd> then <kbd>d</kbd>

4. To reattach to tmux session: ```tmux attach```

#### Get large files from google drive:

```gdown https://drive.google.com/uc?id=<file_id>```