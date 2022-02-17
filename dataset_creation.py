import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import spacy
import neuralcoref
import numpy as np
import pandas as pd
from nltk import download
from nltk.tokenize import word_tokenize
from scipy import spatial

from src.argument_parser import parse_arguments
from src.elasticsearch_bm25 import ElasticSearchBM25
from src.nlp_utils import preprocess, load_glove_model
from src.utils import (
    get_timestamp,
    set_new_index,
    get_corpus,
    get_queries,
    sample_fever_data,
    get_top_1000_passages,
    sample_queries_and_passages,
    get_dataset_from_existing_sample,
)
from src.json_encodings import (
    encode_bm25_dataset_to_json,
    encode_ner_dataset_to_json,
    encode_tf_dataset_to_json,
    encode_sem_sim_dataset_to_json,
    encode_coref_res_dataset_to_json,
    encode_fact_checking_dataset_to_json,
)
from src.dataset_sources import SRC_DATASETS, SRC_PRETRAINED_GLOVE

# set visible devices to -1 since no gpu is needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ner_nlp = en_core_web_sm.load()
# coref_nlp = spacy.load('en')

download("stopwords")
download("punkt")

# def get_dataset_from_existing_sample_ir(sample_path: Path) -> pd.DataFrame:
#     try:
#         passage_query_df = pd.read_csv(sample_path, seo=",")
#     except FileNotFoundError:
#         logging.error(f"File {sample_path} you are trying to load the sample from does not exist.")
#         return
#     fever = ir_datasets.load(SRC_DATASETS[args.source]["dataset_path"])
#     doc_store = fever.docs_store()
#     query_store = fever.queries_store()
#     for qrel in fever.qrels_iter():
#         print(qrel)
#         doc = doc_store.get(qrel.doc_id)
#         query = query_store.get(qrel.query_id)
#     passage_query_df["query"] = passage_query_df.apply(lambda x: query_store.get(x), axis=1)


def write_dataset_to_file(task: str, dataset, set_name: Optional[str] = None) -> None:
    output_filename = (
        SRC_DATASETS[args.source]["short"]
        + f"_{set_name + '_' if set_name is not None else ''}{task}_{args.size}_{args.samples_per_query}_{get_timestamp()}.json"
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
        lambda x: bm25.score(x["query"], document_ids=[x["pid"]])[x["pid"]], axis=1
    )

    bm25.delete_container()

    dataset_dict = encode_bm25_dataset_to_json(dataset_df, SRC_DATASETS[args.source]["long"])

    write_dataset_to_file("bm25", dataset_dict)


def tf_dataset_creation(dataset_df: pd.DataFrame) -> None:
    def calculate_tf(query: str, passage: str):
        pp = preprocess(passage)
        pq = preprocess(query)
        return np.average([len([tk for tk in pp if tk == token]) / len(pp) for token in pq])

    dataset_df["avg_tf"] = dataset_df.apply(
        lambda x: calculate_tf(x["query"], x["passage"]), axis=1
    )

    dataset_dict = encode_tf_dataset_to_json(dataset_df, SRC_DATASETS[args.source]["long"])

    write_dataset_to_file("tf", dataset_dict)


def ner_dataset_creation(dataset_df: pd.DataFrame) -> None:
    # key: pid, value: List[([start, end], label)]
    targets: Dict[str, List[Tuple[List, str]]] = {}

    for _, row in dataset_df.iterrows():
        ner_nlp = spacy.load("en_core_web_sm")
        doc = ner_nlp(row["query"] + " " + row["passage"])
        new_targets_doc = [([X.start, X.end], X.label_) for X in doc.ents]
        targets[str(row["pid"]) + " " + str(row["qid"])] = new_targets_doc

    dataset_dict = encode_ner_dataset_to_json(
        dataset_df, targets, SRC_DATASETS[args.source]["long"]
    )

    write_dataset_to_file("ner", dataset_dict)


def sem_sim_dataset_creation(dataset_df: pd.DataFrame, count_oov_tokens: bool = False) -> None:
    glv_model = load_glove_model(Path(SRC_PRETRAINED_GLOVE))
    # get average embedding for cases when token is not present in glove model
    avg_emb = np.average(np.asarray(list(glv_model.values())), axis=0)
    num_oov_toks = 0
    num_toks = 0

    def calculate_cos_sim(passage: str, query: str):
        nonlocal num_oov_toks
        nonlocal num_toks
        d_toks: List[str] = list(map(str.lower, word_tokenize(passage)))
        q_toks: List[str] = list(map(str.lower, word_tokenize(query)))
        if count_oov_tokens:
            num_oov_toks += len([x for x in d_toks if x not in glv_model])
            num_oov_toks += len([x for x in q_toks if x not in glv_model])
            num_toks += len(d_toks)
            num_toks += len(q_toks)
        g_e_doc = np.asarray([glv_model[x] if x in glv_model else avg_emb for x in d_toks])
        g_e_q = np.asarray([glv_model[x] if x in glv_model else avg_emb for x in q_toks])
        cos_sim = np.zeros((g_e_doc.shape[0], g_e_q.shape[0]))
        for i, doc_e in enumerate(g_e_doc):
            for j, q_e in enumerate(g_e_q):
                cos_sim[i][j] = 1 - spatial.distance.cosine(doc_e, q_e)

        return np.average(cos_sim)

    dataset_df["cos_sim"] = dataset_df.apply(
        lambda x: calculate_cos_sim(x["passage"], x["query"]), axis=1
    )
    if count_oov_tokens:
        print(
            f"Number of tokens: {num_toks}, number of oov tokens: {num_oov_toks}, accounting for {(num_oov_toks / num_toks) * 100:.2f}%"
        )

    # free memory
    del glv_model

    dataset_dict = encode_sem_sim_dataset_to_json(dataset_df, SRC_DATASETS[args.source]["long"])

    write_dataset_to_file("sem_sim", dataset_dict)


def coref_res_dataset_creation(dataset_df: pd.DataFrame, source: str) -> None:
    #                         ref
    # key: pid, value: List[([start, end], [])]
    targets: Dict[str, List[Tuple[List[int], List[List[int]]]]] = {}
    coref_nlp = spacy.load("en_core_web_sm")
    neuralcoref.add_to_pipe(coref_nlp)

    for _, row in dataset_df.iterrows():
        doc = coref_nlp(row["query"] + " " + row["passage"])
        query = coref_nlp(row["query"])
        passage = coref_nlp(row["passage"])
        for cluster in doc._.coref_clusters:
            query_references = []
            for i, reference in enumerate(cluster):
                # only consider those references that appear in the query
                if reference.start < len(query) and reference.end <= len(query):
                    query_references.append([reference.start, reference.end])
                elif query_references:
                    targets[str(row["pid"]) + " " + str(row["qid"])]

        print(doc)

    dataset_dict = encode_coref_res_dataset_to_json(
        dataset_df, targets, SRC_DATASETS[source]["long"]
    )

    write_dataset_to_file("coref_res", dataset_dict)


# def fact_checking_dataset_creation(corpus_dfs, queries_dfs, qrels) -> None:
#     fever = ir_datasets.load(SRC_DATASETS[source]["dataset_path"])
#     doc_store = fever.docs_store()
#     query_df = pd.DataFrame(fever.queries_iter())
#     if args.sample_path is not None:
#         get_dataset_from_existing_sample_ir(fever, doc_store, query_df)
#     else:

#         for qrel in fever.qrels_iter():
#             doc = doc_store.get(qrel.doc_id)
#             query = query_df[query_df["query_id"] == qrel.query_id]
#             print(qrel)


def fact_checking_dataset_creation(corpus_dfs, queries_dfs, qrels) -> None:
    json_res = encode_fact_checking_dataset_to_json(corpus_dfs, queries_dfs, qrels)

    for set_name, dataset in json_res.items():
        write_dataset_to_file("fact_checking", dataset, set_name)


def main(args):
    logging.basicConfig(filename="msmarco.log", filemode="w+", level=logging.INFO)

    if any(x in args.tasks for x in ["bm25", "tf", "semsim", "corefres"]):
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
                corpus_df,
                query_df,
                q_p_top1000_dict,
                args.size,
                args.samples_per_query,
                SRC_DATASETS[args.source]["short"],
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
        if "corefres" in args.tasks:
            coref_res_dataset_creation(dataset_df, args.source)

    if "factchecking" in args.tasks:
        corpus_dfs, queries_dfs, qrels = sample_fever_data(args.split, args.size)
        fact_checking_dataset_creation(corpus_dfs, queries_dfs, qrels)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
