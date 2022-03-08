import logging
from typing import Dict, List, Tuple
import pandas as pd


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
                "targets": [{"label": row["bm25"]}],
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
                "targets": [{"label": row["avg_tf"]}],
            }
        )
    logging.info("TF dataset encoded to json.")

    return dataset


def encode_ner_dataset_to_json(
    df: pd.DataFrame, targets: Dict[str, List[Tuple[List, str]]], source: str
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
                "targets": [{"label": row["cos_sim"]}],
            }
        )
    logging.info("Semantic similarity dataset encoded to json.")

    return dataset


def encode_coref_res_dataset_to_json(df: pd.DataFrame, targets, source: str) -> List[Dict]:
    dataset: List[Dict] = []
    for idx, row in df.iterrows():
        # only write sentence to dataset when there exists a coref
        if targets[str(row["pid"]) + " " + str(row["qid"])]:
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
                        {
                            "label": label,
                            "span1": start_end1,
                            "text1": text1,
                            "span2": start_end2,
                            "text2": text2,
                        }
                        for label, text1, start_end1, text2, start_end2 in targets[
                            str(row["pid"]) + " " + str(row["qid"])
                        ]
                    ],
                }
            )
    logging.info("Coreference Resolution dataset encoded to json.")

    return dataset


def encode_fact_checking_dataset_to_json(corpus_dfs, queries_dfs, qrels) -> Dict[str, List[Dict]]:
    json_res: Dict[str, List[Dict]] = {}
    for set_name, qrel in qrels.items():
        dataset = []
        queries = queries_dfs[set_name]
        passages = corpus_dfs[set_name]
        for _, row in qrel.iterrows():
            q_data = queries[queries["_id"] == row["query-id"]]
            p_data = passages[passages["_id"] == row["corpus-id"]]
            dataset.append(
                {
                    "info": {"pid": row["corpus-id"], "qid": row["query-id"], "source": "fever"},
                    "text": q_data.text.values[0] + " [SEP] " + p_data.text.values[0],
                    "input": {"passage": p_data.text.values[0], "query": q_data.text.values[0]},
                    "targets": [{"label": q_data.metadata.values[0]["label"]}],
                }
            )
        json_res[set_name] = dataset
    logging.info("Fact checking dataset encoded to json.")

    return json_res
