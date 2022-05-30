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
Start elasticsearch container in another Terminal:
```sh
docker run -p 12375:9200 -p 12376:9300 -e "discovery.type=single-node" --detach --name es -v esdata1:/usr/share/elasticsearch/data:rw  docker.elastic.co/elasticsearch/elasticsearch:7.16.2
```

Once created you only need to make sure the container is running before you call the script to create datasets.
If it is not running start it via:
```sh
docker container start es
```

To finally run the dataset creation script:
```sh
conda activate tinse
python dataset_creation.py -t=<tasks> -s=<size> -sq=<samples_per_query> ...
```

Naming of Datasets saved in ``/datasets``: ``{source}_{task}_{size}_{samples_per_query}_{timestamp}.json``


#### Options
| Option      | Description | Default  |
| ----------- | ----------- | ----------- |
| -s, --size      | Size of the generated dataset(s) | 10000|
| -sq, --samples_per_query  | Determines the maximumn number of passage samples with the same query in the generated dataset         |  5 |
| -src, --source   | Source Dataset  | msmarco |
| -t, --tasks   | Tasks to generate datasets for. Possible tasks are: ['bm25', 'tf', 'semsim', 'ner', 'corefres', 'factchecking']. Should be comma seperated  | bm25,semsim,ner,tf |
| -sp, --sample_path   | Reuse an existing sample of a dataset. You need to specify the name of the file in ./datasets/samples/. Every time a dataset is newly sampled it is saved in csv format. Naming format: {src}_{size}_{samples per query}_{timestamp}.csv. If set --size and --samples_per_query are ignored) | -|
| -ph, --port_http   | Http Port for elasticsearch container, should correspond to the port the docker container is bound to | 12375 |
| -pt, --port_tcp   | TCP Port for elasticsearch container, should correspond to the port the docker container is bound to | 12376 |
| --split   | Only relevant for  Train, val and test split ratio. Must add up to 100 | 70,15,15 |
| --neg_sample_ratio   | Only relevant for coreference resolution. Ratio of negative sampling containing easy and hard examples. First number corresponds to percentage * 100 of easy examples (random word from the passage), second for harder (other entities in the passage). Must add up to 100 | 50,50 |
| --id_pairs   | Flag whether ID pairs (MSMARCO) from csv (assets/msmarco/passage_re_ranking/msm_id_pairs.csv) should be used to create the datasets  | - |

***

## Directory Contents

### assets
source datasets (msmarco, fever, glove)
Mappings from key to files are made in src/dataset_sources.py.
If another dataset should be added one should follow this schema.

### datasets
generated datasets in json format

### datasets/samples
generated samples to construct datasets which use the same query/document pairs

***

## Other useful commands

#### Run detached script on server:

1. ```tmux```

2. run script

3. To quit tmux session: <kbd>Ctrl</kbd> + <kbd>b</kbd> then <kbd>d</kbd>

4. To reattach to tmux session: ```tmux attach```