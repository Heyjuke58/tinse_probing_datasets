from numpy import int64
import pandas as pd
import time
import logging
import ftfy
from pathlib import Path
import json
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from src.dataset_sources import SRC_FEVER
from collections import defaultdict


def set_new_index(df: pd.DataFrame) -> pd.DataFrame:
    df["idx"] = pd.Index(range(df.shape[0]), dtype=int64)
    return df.set_index("idx")


def get_timestamp() -> str:
    return time.strftime("%Y_%m_%d-%H-%M-%S")


def get_top_1000_passages(path: Path) -> Dict[int, List[int]]:
    # q_id -> [p_id1, p_id2, .. , p_id1000]
    q_p_top1000_dict: Dict[int, List[int]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            q_id, p_id = tuple(line.split(sep="\t")[:2])
            q_p_top1000_dict[int(q_id)].append(int(p_id))
    logging.info("Top 1000 passages per query parsed.")

    return q_p_top1000_dict


def get_corpus(path: Path, fix_unicode_errors: bool = True) -> pd.DataFrame:
    corpus_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["pid", "passage"],
        encoding="utf8",
        dtype={"pid": "int64", "passage": "string"},
    )
    # fix unicode errors
    if fix_unicode_errors:
        corpus_df["passage"] = corpus_df["passage"].apply(ftfy.fix_text)
    logging.info("Corpus preprocessed.")

    return corpus_df


def get_queries(path: Path, fix_unicode_errors: bool = True) -> pd.DataFrame:
    queries_df = pd.read_csv(path, sep="\t", header=None, names=["qid", "query"], encoding="utf8")
    # fix unicode errors
    if fix_unicode_errors:
        queries_df["query"] = queries_df["query"].apply(ftfy.fix_text)
    logging.info("Queries preprocessed.")

    return queries_df


def sample_fever_data(
    split: str, size: int, seed: int = 12
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    split_sizes = [
        round(size * (split_size / 100)) for split_size in list(map(int, split.split(",")))
    ]
    qrels: Dict[str, pd.DataFrame] = {
        f"{set_name}": pd.read_csv(
            Path(SRC_FEVER[f"path_qrels_{set_name}"]), sep="\t", encoding="utf8"
        ).sample(n=size, random_state=seed)
        for set_name, size in zip(["train", "val", "test"], split_sizes)
    }

    unique_docs = {
        f"{set_name}": qrels[set_name]["corpus-id"].unique()
        for set_name in ["test", "val", "train"]
    }
    unique_queries = {
        f"{set_name}": qrels[set_name]["query-id"].unique() for set_name in ["test", "val", "train"]
    }

    corpus: Dict[str, List[Dict]] = defaultdict(list)
    queries: Dict[str, List[Dict]] = defaultdict(list)

    with open(SRC_FEVER["path_corpus"], "r") as corpus_file:
        for line in corpus_file:
            doc = json.loads(line)
            if doc["_id"] in unique_docs["train"]:
                corpus["train"].append(doc)
            if doc["_id"] in unique_docs["val"]:
                corpus["val"].append(doc)
            if doc["_id"] in unique_docs["test"]:
                corpus["test"].append(doc)
    corpus_dfs: Dict[str, pd.DataFrame] = {
        set_name: pd.DataFrame(corpus[set_name]) for set_name in corpus.keys()
    }

    with open(SRC_FEVER["path_queries"], "r") as queries_file:
        for line in queries_file:
            query = json.loads(line)
            if int(query["_id"]) in unique_queries["train"]:
                queries["train"].append(query)
            if int(query["_id"]) in unique_queries["val"]:
                queries["val"].append(query)
            if int(query["_id"]) in unique_queries["test"]:
                queries["test"].append(query)
    queries_dfs: Dict[str, pd.DataFrame] = {
        set_name: pd.DataFrame(queries[set_name]) for set_name in queries.keys()
    }
    for df in queries_dfs.values():
        df["_id"] = pd.to_numeric(df["_id"])

    logging.info("Fever dataset preprocessed.")
    return corpus_dfs, queries_dfs, qrels


def sample_queries_and_passages(
    corpus_df: pd.DataFrame,
    query_df: pd.DataFrame,
    q_p_top1000: Dict[int, List[int]],
    size: int,
    samples_per_query: int,
    source: str,
    save_sample: bool = True,
) -> pd.DataFrame:
    rand_queries: pd.DataFrame = pd.DataFrame()
    rand_passages: pd.DataFrame = pd.DataFrame()

    sampled_queries = random.sample(list(q_p_top1000), len(q_p_top1000.keys()))

    for qid in sampled_queries:
        possible_passages = q_p_top1000[qid]
        sample_size = (
            samples_per_query
            if len(possible_passages) >= samples_per_query
            else len(possible_passages)
        )
        sampled_passages = random.sample(possible_passages, sample_size)
        preprocessed_sampled_passages = corpus_df.loc[corpus_df["pid"].isin(sampled_passages)]
        rand_passages = pd.concat([rand_passages, preprocessed_sampled_passages])
        preprocessed_sampled_queries = query_df.loc[query_df["qid"] == qid]
        if len(preprocessed_sampled_passages) != 0:
            to_concat = pd.concat(
                [preprocessed_sampled_queries] * len(preprocessed_sampled_passages),
                ignore_index=True,
            )
            rand_queries = pd.concat([rand_queries, to_concat], ignore_index=True)

        if len(rand_passages) >= size:
            rand_passages = rand_passages[:size]
            rand_queries = rand_queries[:size]
            break

    assert (
        len(rand_passages) == size
    ), f"Dataset cannot have specified size of {size}. Please lower the value to at least {len(rand_passages)} or increase the maximumn number of samples per query."

    # reset index to merge with queries
    rand_passages = set_new_index(rand_passages)
    rand_queries = set_new_index(rand_queries)

    # concat passages and queries dataframes
    passage_query_df = pd.concat([rand_passages, rand_queries], sort=False, axis=1)
    logging.info(
        f"Dataset sample of size {size} with max samples per query of {samples_per_query} generated."
    )

    if save_sample:
        # Mkdir if it does not exist
        Path("./datasets/samples").mkdir(parents=True, exist_ok=True)

        # save sample
        passage_query_df.to_csv(
            f"./datasets/samples/{source}_{size}_{samples_per_query}_{get_timestamp()}.csv",
            columns=["qid", "pid"],
        )

    return passage_query_df


def get_dataset_from_existing_sample(
    corpus_df: pd.DataFrame, query_df: pd.DataFrame, sample_path: Path
) -> pd.DataFrame:
    try:
        passage_query_df = pd.read_csv(sample_path, sep=",")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File {sample_path} you are trying to load the sample from does not exist."
        )
    passage_query_df["query"] = passage_query_df.apply(
        lambda x: query_df.loc[query_df["qid"] == x["qid"]]["query"].values[0], axis=1
    )
    passage_query_df["passage"] = passage_query_df.apply(
        lambda x: corpus_df.loc[corpus_df["pid"] == x["pid"]]["passage"].values[0], axis=1
    )

    return passage_query_df
