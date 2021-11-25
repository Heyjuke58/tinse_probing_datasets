from typing import List, Dict
from pathlib import Path
import argparse
from contextlib import contextmanager
from time import perf_counter
import json
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

download("stopwords")

STOPWORDS = set(stopwords.words("english"))
POTERT_STEMMER = PorterStemmer()

# PATH_CORPUS = './assets/msmarco/passage_re_ranking/collection.tsv'
PATH_CORPUS = (
    "./assets/msmarco/passage_re_ranking/collection_sample.tsv"
)  # TODO: later change to real corpus (constant above)
PATH_QUERIES = (
    "./assets/msmarco/passage_re_ranking/queries.dev.small.tsv"
)  # TODO: later change to be not small but queries.dev.tsv

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--size",
    type=int,
    dest="size",
    default=5000,
    help="size of the generated dataset",
)
parser.add_argument(
    "-pc",
    "--path_corpus",
    type=str,
    dest="path_corpus",
    default=PATH_CORPUS,
    help="path to the corpus",
)
parser.add_argument(
    "-pq",
    "--path_queires",
    type=str,
    dest="path_queries",
    default=PATH_QUERIES,
    help="path to the queries",
)
args = parser.parse_args()


@contextmanager
def timing(description: str) -> None:
    start = perf_counter()
    yield
    ellapsed_time = perf_counter() - start
    print(f"{description}: {ellapsed_time}")


def tokenize_corpus(path: Path) -> pd.DataFrame:
    with timing("reading entire corpus"):
        corpus_df = pd.read_csv(path, sep="\t", header=None, names=["pid", "passage"])
    with timing("tokenization"):
        corpus_df["preprocessed_passage"] = corpus_df["passage"].apply(word_tokenize)
    with timing("remove punctuation"):
        corpus_df["preprocessed_passage"] = corpus_df["preprocessed_passage"].apply(
            lambda x: [word for word in x if word.isalnum()]
        )
    with timing("filter stopwords"):
        corpus_df["preprocessed_passage"] = corpus_df["preprocessed_passage"].apply(
            lambda x: [word for word in x if not word in STOPWORDS]
        )
    with timing("stemming"):
        corpus_df["preprocessed_passage"] = corpus_df["preprocessed_passage"].apply(
            lambda x: [POTERT_STEMMER.stem(word) for word in x]
        )

    return corpus_df


def tokenize_queries(path: Path) -> pd.DataFrame:
    queries_df = pd.read_csv(path, sep="\t", header=None, names=["qid", "query"])
    queries_df["preprpcessed_query"] = queries_df["query"].apply(word_tokenize)
    queries_df["preprpcessed_query"] = queries_df["preprpcessed_query"].apply(
        lambda x: [word for word in x if word.isalnum()]
    )
    queries_df["preprpcessed_query"] = queries_df["preprpcessed_query"].apply(
        lambda x: [word for word in x if not word in STOPWORDS]
    )
    queries_df["preprpcessed_query"] = queries_df["preprpcessed_query"].apply(
        lambda x: [POTERT_STEMMER.stem(word) for word in x]
    )

    return queries_df

def set_new_index(df: DataFrame) -> DataFrame:
    df['idx'] = pd.Int64Index(range(df.shape[0]))
    return df.set_index('idx')


def build_dataset_from_dataframe(df: DataFrame, source: str) -> List[Dict]:
    dataset: List[Dict] = []
    for index, row in df.iterrows():
        dataset.append({
            'info': {
                'pid': row['pid'], # passage id
                'qid': row['qid'], # query id
                'source': source
            },
            'input': {
                'document': row['passage'],
                'query': row['query']
            },
            'target': row['bm25']
        })
    
    return dataset

def write_dataset_to_file(path: Path, dataset) -> None:
    with open(path, 'w') as outfile:
        json.dump(dataset, outfile, indent=4)


def main(size: int, path_corpus: str, path_queries: str):
    corpus_df = tokenize_corpus(
        Path(path_corpus)
    )  # preprocess corpus (passages) for bm25
    bm25 = BM25Okapi(list(corpus_df["preprocessed_passage"]))
    queries_df = tokenize_queries(Path(path_queries))  # preprocess quueries for bm25

    # get random passages
    rand_idx_passages = np.random.default_rng(seed=0).choice(corpus_df.shape[0], size=size) # TODO: remove seed
    rand_passages = corpus_df.iloc[rand_idx_passages]
    # reset index to merge with queries later
    rand_passages = set_new_index(rand_passages)

    # get random queries
    assert (
        queries_df.shape[0] > size
    ), f"The supposed dataset of size {size} is attempted to sample from {queries_df.shape[0]} queries. Choose a smaller dataset size or increase the number of queries sampled from"

    rand_idx_queries = np.random.default_rng(seed=0).choice( # TODO: remove seed
        queries_df.shape[0], size=size, replace=False
    )
    rand_queries = queries_df.iloc[rand_idx_queries]
    rand_queries = set_new_index(rand_queries)

    # concat passages and queries dataframes
    passage_query_df = pd.concat([rand_passages, rand_queries], sort=False, axis=1)

    # calculate bm25 scores
    passage_query_df['bm25'] = passage_query_df.apply(lambda x: bm25.get_scores(x['preprpcessed_query'])[x['pid']], axis=1)

    
    dataset_dict = build_dataset_from_dataframe(passage_query_df, source='msmarco passage re-ranking')

    write_dataset_to_file(Path('./datasets/msmarco_test.json'), dataset_dict)


if __name__ == "__main__":
    main(args.size, args.path_corpus, args.path_queries)
