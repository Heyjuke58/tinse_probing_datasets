from typing import List, Dict, Tuple
import random
from collections import defaultdict
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

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
ner_nlp = en_core_web_sm.load()


download("stopwords")
download("punkt")

STOPWORDS = set(stopwords.words("english"))
POTERT_STEMMER = PorterStemmer()

PATH_CORPUS = './assets/msmarco/passage_re_ranking/collection_cleaned.tsv'
# PATH_CORPUS = ("./assets/msmarco/passage_re_ranking/collection_sample.tsv") 
PATH_QUERIES = ("./assets/msmarco/passage_re_ranking/queries.dev.tsv")
PATH_TOP1000 = "./assets/msmarco/passage_re_ranking/top1000.dev"


SRC_MS_MARCO = "msmarco passage re-ranking"
OUT_MS_MARCO = "msmarco_bm25_dataset.json"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--size",
    type=int,
    dest="size",
    default=100,
    help="Size of the generated dataset.",
)
parser.add_argument(
    "-sq",
    "--max_samples_per_query",
    type=int,
    dest="max_samples_per_query",
    default=1000,
    help="Determines the maximumn number of passage samples with the same query in the generated dataset.",
)
parser.add_argument(
    "-pc",
    "--path_corpus",
    type=str,
    dest="path_corpus",
    default=PATH_CORPUS,
    help="path to the corpus file",
)
parser.add_argument(
    "-pq",
    "--path_queires",
    type=str,
    dest="path_queries",
    default=PATH_QUERIES,
    help="path to the query file",
)
parser.add_argument(
    "-src",
    "--source",
    type=str,
    dest="source",
    default=SRC_MS_MARCO,
    help="source to add in the info of the dataset",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    dest="output_filename",
    default=OUT_MS_MARCO,
    help="Output filename of the generated dataset"
)
args = parser.parse_args()


@contextmanager
def timing(description: str) -> None:
    start = perf_counter()
    yield
    ellapsed_time = perf_counter() - start
    print(f"{description}: {ellapsed_time}")


def tokenize_corpus(path: Path) -> pd.DataFrame:
    corpus_df = pd.read_csv(path, sep="\t", header=None, names=["pid", "passage"], encoding='utf-8')
    corpus_df["preprocessed_passage"] = corpus_df["passage"].apply(word_tokenize)
    corpus_df["preprocessed_passage"] = corpus_df["preprocessed_passage"].apply(
        lambda x: [word for word in x if word.isalnum()]
    )
    corpus_df["preprocessed_passage"] = corpus_df["preprocessed_passage"].apply(
        lambda x: [word for word in x if not word in STOPWORDS]
    )
    corpus_df["preprocessed_passage"] = corpus_df["preprocessed_passage"].apply(
        lambda x: [POTERT_STEMMER.stem(word) for word in x]
    )

    return corpus_df


def tokenize_queries(path: Path) -> pd.DataFrame:
    queries_df = pd.read_csv(path, sep="\t", header=None, names=["qid", "query"])
    queries_df["preprocessed_query"] = queries_df["query"].apply(word_tokenize)
    queries_df["preprocessed_query"] = queries_df["preprocessed_query"].apply(
        lambda x: [word for word in x if word.isalnum()]
    )
    queries_df["preprocessed_query"] = queries_df["preprocessed_query"].apply(
        lambda x: [word for word in x if not word in STOPWORDS]
    )
    queries_df["preprocessed_query"] = queries_df["preprocessed_query"].apply(
        lambda x: [POTERT_STEMMER.stem(word) for word in x]
    )

    return queries_df

def get_top_1000_passages(path: Path) -> Dict[int, List[int]]:
    # not all queries have 1000 associated passages (sometimes only 1)!
    # df = pd.read_csv(path, sep="\t", header=None, names=["qid", "pid"], usecols=[0, 1])

    # sanity check
    # unique_qids = df['qid'].unique()
    # too_less = []
    # for unique_qid in np.nditer(unique_qids):
    #     filtered_df = df.loc[df["qid"] == unique_qid]
    #     if filtered_df.shape[0] != 1000:
    #         too_less.append(filtered_df.shape[0])
    # print(min(too_less))

    q_p_top1000_dict: Dict[int, List[int]] = defaultdict(list) # q_id -> [p_id1, p_id2, .. , p_id1000]
    with open(path, 'r') as f:
        for line in f:
            q_id, p_id = tuple(line.split(sep="\t")[:2])
            q_p_top1000_dict[int(q_id)].append(int(p_id))
    
    return q_p_top1000_dict

def set_new_index(df: DataFrame) -> DataFrame:
    df["idx"] = pd.Int64Index(range(df.shape[0]))
    return df.set_index("idx")


def build_bm25_dataset_from_dataframe(df: DataFrame, source: str) -> List[Dict]:
    dataset: List[Dict] = []
    for idx, row in df.iterrows():
        dataset.append(
            {
                "info": {
                    "pid": row["pid"],  # passage id
                    "qid": row["qid"],  # query id
                    "source": source,
                },
                "text": row["query"] + " [SEP] " + row["passage"],
                "input": {"passage": row["passage"], "query": row["query"]},
                "target": row["bm25"],
            }
        )

    return dataset


def build_ner_dataset_from_dataframe(df: DataFrame, df_targets: DataFrame, source: str) -> List[Dict]:
    dataset: List[Dict] = []
    for idx, row in df.iterrows():
        dataset.append(
            {
                "info": {
                    "pid": row["pid"],  # passage id
                    "qid": row["qid"],  # query id
                    "source": source,
                },
                "text": row["query"] + " [SEP] " + row["passage"],
                "input": {"passage": row["passage"], "query": row["query"]},
                "targets": [
                    {"span1": [row2["start"], row2["end"]], "label": row2["label"]} 
                    for idx2, row2 in df_targets[df_targets["pid"] == row["pid"]].iterrows()
                ],
            }
        )

    return dataset 


def build_sem_sim_dataset_from_dataframe(df: DataFrame, source: str) -> List[Dict]:
    dataset: List[Dict] = []
    for idx, row in df.iterrows():
        dataset.append(
            {
                "info": {
                    "pid": row["pid"],  # passage id
                    "qid": row["qid"],  # query id
                    "source": source,
                },
                "text": row["query"] + " [SEP] " + row["passage"],
                "input": {"passage": row["passage"], "query": row["query"]},
                "target": row["cos_sim"],
            }
        )

    return dataset 


def write_dataset_to_file(path: Path, dataset) -> None:
    with open(path, "w", encoding='utf8') as outfile:
        json.dump(dataset, outfile, indent=4, ensure_ascii=False)


def bm25_dataset_creation(size: int, max_samples_per_query: int, corpus_df: DataFrame, queries_df: DataFrame, source: str, output_filename: str) -> None:
    bm25 = BM25Okapi(list(corpus_df["preprocessed_passage"]))
    q_p_top1000 = get_top_1000_passages(PATH_TOP1000) # get dict of the top 1000 passages per query

    rand_queries = pd.DataFrame()
    rand_passages = pd.DataFrame()
    
    sampled_queries = random.sample(list(q_p_top1000), len(q_p_top1000.keys()))
    
    for qid in sampled_queries:
        possible_passages = q_p_top1000[qid]
        sample_size = max_samples_per_query if len(possible_passages) >= max_samples_per_query else len(possible_passages)
        sampled_passages = random.sample(possible_passages, sample_size)
        preprocessed_sampled_passages = corpus_df.loc[corpus_df['pid'].isin(sampled_passages)]
        rand_passages = rand_passages.append(preprocessed_sampled_passages)
        preprocessed_sampled_queries = queries_df.loc[queries_df['qid'] == qid]
        rand_queries = rand_queries.append([preprocessed_sampled_queries] * len(preprocessed_sampled_passages), ignore_index=True)
        
        if len(rand_passages) >= size:
            rand_passages = rand_passages[:size]
            rand_queries = rand_queries[:size]
            break
    
    assert len(rand_passages) == size, f"Dataset cannot have specified size of {size}. Please lower the value to at least {len(rand_passages)} or increase the maximumn number of samples per query."


    # reset index to merge with queries
    rand_passages = set_new_index(rand_passages)
    rand_queries = set_new_index(rand_queries)

    # concat passages and queries dataframes
    passage_query_df = pd.concat([rand_passages, rand_queries], sort=False, axis=1)

    # calculate bm25 scores
    passage_query_df["bm25"] = passage_query_df.apply(
        lambda x: bm25.get_scores(x["preprocessed_query"])[x["pid"]], axis=1
    )

    dataset_dict = build_bm25_dataset_from_dataframe(passage_query_df, source)

    write_dataset_to_file(Path("./datasets/") / output_filename, dataset_dict)


def ner_dataset_creation(size: int, max_samples_per_query: int, corpus_df: DataFrame, source: str, output_filename: str) -> None:
    df_passages: Dict[int, str] = {}
    df_targets: List[Tuple[int, List, str]] = [] # list of (pid, [start, end], label)

    # sample passages for the dataset
    sampled_pids = corpus_df["pid"].sample(size)

    # extract named entities
    for pid in sampled_pids:
        passage = corpus_df.loc[pid].passage
        df_passages[pid] = passage
        doc = ner_nlp(passage)
        for X in doc.ents:
            start = passage.find(X.test)
            df_targets.append()
        df_targets.extend([(pid, [passage.find(X.text), ], X.label) for X in doc.ents])
        print([(X.text, X.label_) for X in doc.ents])

    dataset_dict = build_ner_dataset_from_dataframe(df, df_targets)
    write_dataset_to_file(Path("./dataset"), output_filename, dataset_dict)


def sem_sim_dataset_creation(size: int, max_samples_per_query: int, corpus_df: DataFrame, queries_df: DataFrame, source: str, output_filename: str) -> None:


    dataset_dict = build_sem_sim_dataset_from_dataframe(df, df_targets)
    write_dataset_to_file(Path("./dataset"), output_filename, dataset_dict)



if __name__ == "__main__":
    corpus_df = tokenize_corpus(Path(args.path_corpus))  # preprocess corpus (passages) for bm25
    queries_df = tokenize_queries(Path(args.path_queries))  # preprocess quueries for bm25
    
    bm25_dataset_creation(
        args.size,
        args.max_samples_per_query,
        corpus_df,
        queries_df,
        args.source, 
        args.output_filename
    )
    # ner_dataset_creation(
    #     args.size,
    #     args.max_samples_per_query,
    #     corpus_df,
    #     args.source,
    #     args.output_filename
    # )
