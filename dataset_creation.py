import argparse
import logging
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import os

import pandas as pd
import numpy as np
from scipy import spatial
import en_core_web_sm
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ftfy
from src.ElasticSearchBM25 import ElasticSearchBM25
from src.utils import get_timestamp, set_new_index
import ir_datasets

# set visible devices to -1 since no gpu is needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ner_nlp = en_core_web_sm.load()

download("stopwords")
download("punkt")

STOPWORDS = set(stopwords.words("english"))
PORTER_STEMMER = PorterStemmer()

SRC_PRETRAINED_GLOVE = "./assets/glove/glove.6B.300d.txt"

SRC_MS_MARCO = {
    "short": "msmarco",
    "long": "msmarco passage re-ranking",
    "index_name": "msmarco3",
    # "path_corpus": "./assets/msmarco/passage_re_ranking/collection_sample_orig.tsv",
    "path_corpus": "./assets/msmarco/passage_re_ranking/collection.tsv",
    "path_queries": "./assets/msmarco/passage_re_ranking/queries.dev.tsv",
    # "path_queries": "./assets/msmarco/passage_re_ranking/queries.dev.small.tsv",
    "path_top1000": "./assets/msmarco/passage_re_ranking/top1000.dev",
}

SRC_TREC = {}
SRC_FEVER = {"dataset_path": "beir/fever/test"}

SRC_DATASETS = {"msmarco": SRC_MS_MARCO, "trec": SRC_TREC, "fever": SRC_FEVER}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--size", type=int, dest="size", default=10000, help="Size of the generated dataset.",
)
parser.add_argument(
    "-sq",
    "--samples_per_query",
    type=int,
    dest="samples_per_query",
    default=5,
    help="Determines the maximumn number of passage samples with the same query in the generated dataset.",
)
parser.add_argument(
    "-src",
    "--source",
    type=str,
    dest="source",
    default="msmarco",
    help="Source to add in the info of the dataset",
)
parser.add_argument(
    "-ph",
    "--port_http",
    type=str,
    dest="port_http",
    default="12375",
    help="Http port for elasticsearch container",
)
parser.add_argument(
    "-pt",
    "--port_tcp",
    type=str,
    dest="port_tcp",
    default="12376",
    help="Tcp port for elasticsearch container",
)
parser.add_argument(
    "-t",
    "--tasks",
    type=str,
    dest="tasks",
    default="bm25,semsim,ner,tf",
    help="Tasks to run. Possible tasks are: ['bm25', 'semsim', 'ner']. Should be comma seperated",
)
parser.add_argument(
    "-sp",
    "--sample_path",
    type=str,
    dest="sample_path",
    default=None,
    help="""Reuse an existing sample of a dataset. You need to specify the name of the file in ./datasets/samples/.
        Every time a dataset is newly sampled it is saved in csv format. Naming format: {src}_{size}_{samples per query}_{timestamp}.csv
        If set --size and --samples_per_query are ignored.""",
)
args = parser.parse_args()


def preprocess(string: str) -> List[str]:
    """Tokenize string, remove stopwords, stemming

    Args:
        string (str): The text to preprocess

    Returns:
        List[str]: The resulting list of tokens
    """
    x = word_tokenize(string)
    x = [word for word in x if word.isalnum()]
    x = [word for word in x if not word in STOPWORDS]
    x = [PORTER_STEMMER.stem(word) for word in x]
    return x


def get_corpus(path: Path) -> pd.DataFrame:
    corpus_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["pid", "passage"],
        encoding="utf8",
        dtype={"pid": "int64", "passage": "string"},
    )
    # fix unicode errors
    corpus_df["passage"] = corpus_df["passage"].apply(ftfy.fix_text)
    logging.info("Corpus preprocessed.")

    return corpus_df


def get_queries(path: Path) -> pd.DataFrame:
    queries_df = pd.read_csv(path, sep="\t", header=None, names=["qid", "query"], encoding="utf8")
    # fix unicode errors
    queries_df["query"] = queries_df["query"].apply(ftfy.fix_text)
    logging.info("Queries preprocessed.")

    return queries_df


def get_top_1000_passages(path: Path) -> Dict[int, List[int]]:
    # q_id -> [p_id1, p_id2, .. , p_id1000]
    q_p_top1000_dict: Dict[int, List[int]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            q_id, p_id = tuple(line.split(sep="\t")[:2])
            q_p_top1000_dict[int(q_id)].append(int(p_id))
    logging.info("Top 1000 passages per query parsed.")

    return q_p_top1000_dict


def get_dataset_from_existing_sample(
    corpus_df: pd.DataFrame, query_df: pd.DataFrame, sample_path: Path
) -> pd.DataFrame:
    try:
        passage_query_df = pd.read_csv(sample_path, sep=",")
    except FileNotFoundError:
        logging.error(f"File {sample_path} you are trying to load the sample from does not exist.")
        return
    passage_query_df["query"] = passage_query_df.apply(
        lambda x: query_df.loc[query_df["qid"] == x["qid"]]["query"].values[0], axis=1
    )
    passage_query_df["passage"] = passage_query_df.apply(
        lambda x: corpus_df.loc[corpus_df["pid"] == x["pid"]]["passage"].values[0], axis=1
    )

    return passage_query_df

def get_dataset_from_existing_sample_ir(sample_path: Path) -> pd.DataFrame:
    try:
        passage_query_df = pd.read_csv(sample_path, seo=",")
    except FileNotFoundError:
        logging.error(f"File {sample_path} you are trying to load the sample from does not exist.")
        return
    fever = ir_datasets.load(SRC_DATASETS[args.source]["dataset_path"])
    doc_store = fever.docs_store()
    query_store = fever.queries_store()
    for qrel in fever.qrels_iter():
        print(qrel)
        doc = doc_store.get(qrel.doc_id)
        query = query_store.get(qrel.query_id)
    passage_query_df["query"] = passage_query_df.apply(
        lambda x: query_store.get(x), axis=1
    )



def sample_queries_and_passages(
    corpus_df: pd.DataFrame,
    query_df: pd.DataFrame,
    q_p_top1000: Dict[int, List[int]],
    size: int,
    samples_per_query: int,
    save_sample: bool = True,
) -> pd.DataFrame:
    rand_queries = pd.DataFrame()
    rand_passages = pd.DataFrame()

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
        rand_passages = rand_passages.append(preprocessed_sampled_passages)
        preprocessed_sampled_queries = query_df.loc[query_df["qid"] == qid]
        rand_queries = rand_queries.append(
            [preprocessed_sampled_queries] * len(preprocessed_sampled_passages), ignore_index=True,
        )

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
            f"./datasets/samples/{SRC_DATASETS[args.source]['short']}_{size}_{samples_per_query}_{get_timestamp()}.csv",
            columns=["qid", "pid"],
        )

    return passage_query_df


def load_glove_model(glove_file: Path) -> np.ndarray:
    logging.info("Loading Glove Model")
    glove_model = {}
    with open(glove_file, "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    logging.info(f"{len(glove_model)} words loaded!")

    return glove_model


def encode_bm25_dataset_to_json(df: pd.DataFrame, source: str) -> List[Dict]:
    dataset: List[Dict] = []
    for _, row in df.iterrows():
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
    logging.info("BM25 dataset encoded to json.")

    return dataset


def encode_tf_dataset_to_json(df: pd.DataFrame, source: str) -> List[Dict]:
    dataset: List[Dict] = []
    for _, row in df.iterrows():
        dataset.append(
            {
                "info": {
                    "pid": row["pid"],  # passage id
                    "qid": row["qid"],  # query id
                    "source": source,
                },
                "text": row["query"] + " [SEP] " + row["passage"],
                "input": {"passage": row["passage"], "query": row["query"]},
                "target": row["avg_tf"],
            }
        )
    logging.info("TF dataset encoded to json.")

    return dataset


def encode_ner_dataset_to_json(
    df: pd.DataFrame, targets: Dict[str, List[Tuple[List, str]]], source: str,
) -> List[Dict]:
    dataset: List[Dict] = []
    for _, row in df.iterrows():
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
                    {"span1": start_end, "label": label}
                    for start_end, label in targets[str(row["pid"]) + " " + str(row["qid"])]
                ],
            }
        )
    logging.info("NER dataset encoded to json.")

    return dataset


def encode_sem_sim_dataset_to_json(df: pd.DataFrame, source: str) -> List[Dict]:
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
    logging.info("Semantic similarity dataset encoded to json.")

    return dataset


def write_dataset_to_file(task: str, dataset) -> None:
    output_filename = (
        SRC_DATASETS[args.source]["short"]
        + f"_{task}_{args.size}_{args.samples_per_query}_{get_timestamp()}.json"
    )
    path = Path("./datasets") / output_filename
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(dataset, outfile, indent=4, ensure_ascii=False)

    logging.info(f"{task} dataset saved to ./datasets/{output_filename}")


def bm25_dataset_creation(dataset_df: pd.DataFrame, corpus_df: pd.DataFrame) -> None:
    pool = corpus_df["passage"].to_dict()

    bm25 = ElasticSearchBM25(
        pool,
        index_name=SRC_DATASETS[args.source]["index_name"],
        service_type="docker",
        max_waiting=100,
        port_http=args.port_http,
        port_tcp=args.port_tcp,
        es_version="7.16.2",
        reindexing=False,
    )

    # free memory
    del pool
    del corpus_df

    # calculate bm25 scores
    dataset_df["bm25"] = dataset_df.apply(
        lambda x: bm25.score(x["query"], document_ids=[x["pid"]])[x["pid"]], axis=1,
    )

    bm25.delete_container()

    dataset_dict = encode_bm25_dataset_to_json(dataset_df, SRC_DATASETS[args.source]["long"])

    write_dataset_to_file("bm25", dataset_dict)


def tf_dataset_creation(dataset_df: pd.DataFrame) -> None:
    def calculate_tf(query: str, passage: str) -> float:
        pp = preprocess(passage)
        pq = preprocess(query)
        return np.average([len([tk for tk in pp if tk == token]) / len(pp) for token in pq])

    dataset_df["avg_tf"] = dataset_df.apply(
        lambda x: calculate_tf(x["query"], x["passage"]), axis=1,
    )

    dataset_dict = encode_tf_dataset_to_json(dataset_df, SRC_DATASETS[args.source]["long"])

    write_dataset_to_file("tf", dataset_dict)


def ner_dataset_creation(dataset_df: pd.DataFrame) -> None:
    # key: pid, value: List[([start, end], label)]
    targets: Dict[str, List[Tuple[List, str]]] = {}

    for _, row in dataset_df.iterrows():
        doc = ner_nlp(row["query"] + " " + row["passage"])
        new_targets_doc = [([X.start, X.end], X.label_) for X in doc.ents]
        targets[str(row["pid"]) + " " + str(row["qid"])] = new_targets_doc

    dataset_dict = encode_ner_dataset_to_json(
        dataset_df, targets, SRC_DATASETS[args.source]["long"]
    )

    write_dataset_to_file("ner", dataset_dict)


def sem_sim_dataset_creation(dataset_df: pd.DataFrame, count_oov_tokens: bool=False) -> None:
    glove_model = load_glove_model(Path(SRC_PRETRAINED_GLOVE))
    # get average embedding for cases when token is not present in glove model
    avg_embedding = np.average(np.asarray(list(glove_model.values())), axis=0)
    num_oov_tokens = 0
    num_tokens = 0
    def calculate_cos_sim(passage: str, query: str) -> float:
        nonlocal num_oov_tokens
        nonlocal num_tokens
        doc = list(map(str.lower, word_tokenize(passage)))
        query = list(map(str.lower, word_tokenize(query)))
        if count_oov_tokens:
            num_oov_tokens += len([x for x in doc if x not in glove_model])
            num_oov_tokens += len([x for x in query if x not in glove_model])
            num_tokens += len(doc)
            num_tokens += len(query)
        g_e_doc = np.asarray([glove_model[x] if x in glove_model else avg_embedding for x in doc])
        g_e_q = np.asarray([glove_model[x] if x in glove_model else avg_embedding for x in query])
        cos_sim = np.zeros((g_e_doc.shape[0], g_e_q.shape[0]))
        for i, doc_e in enumerate(g_e_doc):
            for j, q_e in enumerate(g_e_q):
                cos_sim[i][j] = 1 - spatial.distance.cosine(doc_e, q_e)

        return np.average(cos_sim)

    dataset_df["cos_sim"] = dataset_df.apply(
        lambda x: calculate_cos_sim(x["passage"], x["query"]), axis=1
    )
    if count_oov_tokens:
        print(f"Number of tokens: {num_tokens}, number of oov tokens: {num_oov_tokens}, accounting for {(num_oov_tokens / num_tokens) * 100:.2f}%")

    # free memory
    del glove_model

    dataset_dict = encode_sem_sim_dataset_to_json(dataset_df, SRC_DATASETS[args.source]["long"])

    write_dataset_to_file("sem_sim", dataset_dict)

def fact_checking_dataset_creation() -> None:
    fever = ir_datasets.load(SRC_DATASETS[args.source]["dataset_path"])
    doc_store = fever.docs_store()
    query_df = pd.DataFrame(fever.queries_iter())
    if args.sample_path is not None:
        get_dataset_from_existing_sample_ir(fever, doc_store, query_df)
    else:

        for qrel in fever.qrels_iter():
            doc = doc_store.get(qrel.doc_id)
            query = query_df[query_df["query_id"] == qrel.query_id]
            print(qrel)



def main():
    logging.basicConfig(filename="msmarco.log", filemode="w+", level=logging.INFO)

    if any(x in args.tasks for x in ["bm25", "tf", "semsim"]):
        # get corpus (passages) for bm25
        corpus_df = get_corpus(Path(SRC_DATASETS[args.source]["path_corpus"]))
        # get quueries for bm25
        query_df = get_queries(Path(SRC_DATASETS[args.source]["path_queries"]))

        # decide whether to construct datset from existing sample or sample passages and queries newly
        if args.sample_path is not None:
            dataset_df = get_dataset_from_existing_sample(
                corpus_df, query_df, Path("./datasets/samples") / args.sample_path
            )
        else:
            # dict query to relevant passages
            q_p_top1000_dict = get_top_1000_passages(SRC_DATASETS[args.source]["path_top1000"])
            dataset_df = sample_queries_and_passages(
                corpus_df, query_df, q_p_top1000_dict, args.size, args.samples_per_query
            )

            del q_p_top1000_dict

        del query_df

        if "bm25" in args.tasks:
            bm25_dataset_creation(dataset_df, corpus_df)

        del corpus_df

        if "tf" in args.tasks:
            tf_dataset_creation(dataset_df)
        if "ner" in args.tasks:
            ner_dataset_creation(dataset_df)
        if "semsim" in args.tasks:
            sem_sim_dataset_creation(dataset_df, count_oov_tokens=True)

    if "factchecking" in args.tasks:
        fact_checking_dataset_creation()


if __name__ == "__main__":
    main()