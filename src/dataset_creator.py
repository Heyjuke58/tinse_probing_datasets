import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ir_datasets
import neuralcoref
import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from scipy import spatial

from src.dataset_sources import SRC_DATASETS, SRC_PRETRAINED_GLOVE
from src.elasticsearch_bm25 import ElasticSearchBM25
from src.json_encodings import (
    encode_bm25_dataset_to_json,
    encode_coref_res_dataset_to_json,
    encode_fact_checking_dataset_to_json,
    encode_ner_dataset_to_json,
    encode_sem_sim_dataset_to_json,
    encode_tf_dataset_to_json,
)
from src.nlp_utils import load_glove_model, preprocess
from src.utils import (
    get_corpus,
    get_dataset_from_existing_sample,
    get_queries,
    get_timestamp,
    get_top_1000_passages,
    sample_fever_data,
    sample_queries_and_passages,
    set_new_index
)


class DatasetCreator:
    def __init__(
        self,
        tasks: List[str],
        source: str,
        size: int,
        samples_per_query: int,
        port_http: str,
        port_tcp: str,
        sample_path: Optional[str] = None,
        neg_sample_ratio: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        logging.basicConfig(
            filename=f"{source}_{get_timestamp()}.log", filemode="w+", level=logging.INFO
        )

        assert all(
            [task in ["bm25", "tf", "semsim", "ner", "corefres", "factchecking"] for task in tasks]
        ), f"Invalid task in {tasks}"

        self.tasks = tasks
        self.source = source
        self.size = size
        self.samples_per_query = samples_per_query

        self.port_http = port_http
        self.port_tcp = port_tcp

        self.sample_path = sample_path
        self.neg_sample_ratio = neg_sample_ratio
        self.split = split

        self.task_methods = {
            "bm25": self.bm25_dataset_creation,
            "tf": self.tf_dataset_creation,
            "semsim": self.sem_sim_dataset_creation,
            "ner": self.ner_dataset_creation,
            "corefres": self.coref_res_dataset_creation,
            "factchecking": self.fact_checking_dataset_creation,
        }

        if self.sample_path is not None:
            sp = self.sample_path.split("_")
            self.size, self.samples_per_query = int(sp[1]), int(sp[2])

        if any(x in self.tasks for x in ["bm25", "tf", "semsim", "ner"]):
            # get corpus (passages) for bm25
            self.corpus_df = get_corpus(Path(SRC_DATASETS[self.source]["path_corpus"]))
            # get quueries for bm25
            self.query_df = get_queries(Path(SRC_DATASETS[self.source]["path_queries"]))
            self.q_p_top1000_dict = None

            # decide whether to construct datset from existing sample or sample passages and queries newly
            if self.sample_path is not None:
                self.dataset_df = get_dataset_from_existing_sample(
                    self.corpus_df, self.query_df, Path("./datasets/samples") / self.sample_path
                )
            else:
                # dict query to relevant passages
                self.q_p_top1000_dict = get_top_1000_passages(
                    SRC_DATASETS[self.source]["path_top1000"]
                )
                self.dataset_df = sample_queries_and_passages(
                    self.corpus_df,
                    self.query_df,
                    self.q_p_top1000_dict,
                    self.size,
                    self.samples_per_query,
                    SRC_DATASETS[self.source]["short"],
                )

            # free memory if data is not needed anymore
            if not "corefres" in self.tasks:
                del self.q_p_top1000_dict
                del self.query_df

        elif any(x in self.tasks for x in ["corefres"]):
            # dict query to relevant passages
            self.q_p_top1000_dict = get_top_1000_passages(
                SRC_DATASETS[self.source]["path_top1000"]
            )
            self.query_df = get_queries(Path(SRC_DATASETS[self.source]["path_queries"]))


    def run(self):
        """Method to generate datasets for al tasks"""
        for task, func in self.task_methods.items():
            if task in self.tasks:
                func()

    def write_dataset_to_file(self, task: str, dataset, set_name: Optional[str] = None) -> None:
        output_filename = (
            SRC_DATASETS[self.source]["short"]
            + f"_{set_name + '_' if set_name is not None else ''}{task}_{self.size}_{self.samples_per_query}_{get_timestamp()}.json"
        )
        path = Path("./datasets") / output_filename
        with open(path, "w", encoding="utf8") as outfile:
            json.dump(dataset, outfile, indent=4, ensure_ascii=False)

        logging.info(f"{task} dataset saved to ./datasets/{output_filename}")

    def bm25_dataset_creation(self) -> None:
        bm25_df = self.dataset_df.copy()
        pool = self.corpus_df["passage"].to_dict()

        bm25 = ElasticSearchBM25(
            pool,
            index_name=SRC_DATASETS[self.source]["index_name"],
            service_type="docker",
            max_waiting=100,
            port_http=self.port_http,
            port_tcp=self.port_tcp,
            es_version="7.16.2",
            reindexing=False,
        )

        # free memory
        del pool
        del self.corpus_df

        # calculate bm25 scores
        bm25_df["bm25"] = bm25_df.apply(
            lambda x: bm25.score(x["query"], document_ids=[x["pid"]])[x["pid"]], axis=1
        )

        bm25.delete_container()

        dataset_dict = encode_bm25_dataset_to_json(bm25_df, SRC_DATASETS[self.source]["long"])

        self.write_dataset_to_file("bm25", dataset_dict)

    def tf_dataset_creation(self) -> None:
        tf_df = self.dataset_df.copy()

        def calculate_tf(query: str, passage: str):
            pp = preprocess(passage)
            pq = preprocess(query)
            return np.average([len([tk for tk in pp if tk == token]) / len(pp) for token in pq])

        tf_df["avg_tf"] = tf_df.apply(lambda x: calculate_tf(x["query"], x["passage"]), axis=1)

        dataset_dict = encode_tf_dataset_to_json(tf_df, SRC_DATASETS[self.source]["long"])

        self.write_dataset_to_file("tf", dataset_dict)

    def ner_dataset_creation(self) -> None:
        ner_df = self.dataset_df.copy()
        # key: pid, value: List[([start, end], label)]
        targets: Dict[str, List[Tuple[List, str]]] = {}

        for _, row in ner_df.iterrows():
            ner_nlp = spacy.load("en_core_web_sm")
            doc = ner_nlp(row["query"] + " " + row["passage"])
            new_targets_doc = [([X.start, X.end], X.label_) for X in doc.ents]
            targets[str(row["pid"]) + " " + str(row["qid"])] = new_targets_doc

        dataset_dict = encode_ner_dataset_to_json(
            ner_df, targets, SRC_DATASETS[self.source]["long"]
        )

        self.write_dataset_to_file("ner", dataset_dict)

    def sem_sim_dataset_creation(self, count_oov_tokens: bool = False) -> None:
        sem_sim_df = self.dataset_df.copy()
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

        sem_sim_df["cos_sim"] = sem_sim_df.apply(
            lambda x: calculate_cos_sim(x["passage"], x["query"]), axis=1
        )
        if count_oov_tokens:
            print(
                f"Number of tokens: {num_toks}, number of oov tokens: {num_oov_toks}, accounting for {(num_oov_toks / num_toks) * 100:.2f}%"
            )

        # free memory
        del glv_model

        dataset_dict = encode_sem_sim_dataset_to_json(sem_sim_df, SRC_DATASETS[self.source]["long"])

        self.write_dataset_to_file("sem_sim", dataset_dict)

    def coref_res_dataset_creation(self) -> None:
        assert isinstance(
            self.q_p_top1000_dict, dict
        ), "Qrels must be defined for coref res dataset creation"

        # coref_res_df: pd.DataFrame = pd.DataFrame()
        # coref_res_df = self.dataset_df.copy()

        #                         ref
        # key: pid + qid, value: List[(label(True|False), query token(str), [start, end], passage token, [start, end])]
        targets: Dict[str, List[Tuple[bool, str, List, str, List]]] = defaultdict(list)
        coref_nlp = spacy.load("en_core_web_sm")
        neuralcoref.add_to_pipe(coref_nlp)

        assert isinstance(
            self.neg_sample_ratio, str
        ), "Negative sampling ratio has to be defined when creating coref res dataset"
        neg_sample_ratio_easy = float(self.neg_sample_ratio.split(",")[0]) / 100
        num_easy_neg_samples = 0
        total_samples = 0 # for controlling the ratio of hard/easy negative samples
        total_dataset_size = 0 # for controlling the size of the dataset to be created

        def hard_example_possible(doc, query_len, query_ref_text, doc_ref) -> List:
            possible_hard_examples = []
            for ent in doc.ents:
                # continue if entity is in query (doc contains both query and passage)
                if ent.start <= query_len:
                    continue
                # continue if entitiy and query or doc text share a string
                elif (
                    ent.text.lower() in doc_ref.text.lower()
                    or ent.text.lower() in query_ref_text.lower()
                    or doc_ref.text.lower() in ent.text.lower()
                    or query_ref_text.lower() in ent.text.lower()
                ):
                    continue
                # continue if entitity range (start, end) overlaps doc_ref range
                elif ent.start <= doc_ref.end and doc_ref.start <= ent.end:
                    continue
                # otherwise we have a valid hard negative example
                else:
                    possible_hard_examples.append(ent)
            return possible_hard_examples

        sampled_queries = random.sample(
            list(self.q_p_top1000_dict), len(self.q_p_top1000_dict.keys())
        )
        corpus = ir_datasets.load(SRC_DATASETS[self.source]["ir_path"])
        doc_store = corpus.docs_store()

        passages: pd.DataFrame = pd.DataFrame()
        queries: pd.DataFrame = pd.DataFrame()

        for qid in sampled_queries:
            samples_per_query = 0
            possible_passages = self.q_p_top1000_dict[qid]
            sampled_passages = random.sample(possible_passages, len(possible_passages))
            q = self.query_df.loc[self.query_df["qid"] == qid]
            q_text = q["query"].values[0]
            for pid in sampled_passages:
                has_target = False
                p = doc_store.get(str(pid)).text
                doc = coref_nlp(q_text + " " + p)
                query = coref_nlp(q_text)
                for cluster in doc._.coref_clusters:
                    query_references: List[Tuple[str, int, int]] = []
                    for i, reference in enumerate(cluster):
                        # only consider those references that appear in the query
                        if reference.start < len(query) and reference.end <= len(query):
                            query_references.append(
                                (reference.text, reference.start, reference.end)
                            )
                        elif query_references:
                            for text, start, end in query_references:
                                # add positive example
                                if not has_target:
                                    samples_per_query += 1
                                    total_dataset_size += 1
                                    has_target = True
                                    passages = pd.concat([passages, pd.DataFrame({"pid": pid, "passage": p}, index=[pid])])
                                targets[str(pid) + " " + str(qid)].append(
                                    (
                                        True,
                                        text,
                                        [start, end],
                                        reference.text,
                                        [reference.start, reference.end],
                                    )
                                )
                                # add negative example
                                total_samples += 1
                                possible_hard_examples = hard_example_possible(
                                    doc, len(query), text, reference
                                )
                                easy = (
                                    True
                                    if num_easy_neg_samples / total_samples < neg_sample_ratio_easy
                                    # get easy example if hard one is not possiblle
                                    or not possible_hard_examples
                                    else False
                                )
                                if easy:
                                    num_easy_neg_samples += 1
                                    random_word = text
                                    while random_word == text:
                                        random_word = random.sample(set(coref_nlp.vocab.strings), 1)[0]
                                    targets[str(pid) + " " + str(qid)].append(
                                        (False, text, [start, end], random_word, [])
                                    )
                                else:
                                    neg_sample = random.choice(possible_hard_examples)
                                    targets[str(pid) + " " + str(qid)].append(
                                        (
                                            False,
                                            text,
                                            [start, end],
                                            neg_sample.text,
                                            [neg_sample.start, neg_sample.end],
                                        )
                                    )
                # break (and continue with next query) if enough passages have been sampled for a single query
                if samples_per_query >= self.samples_per_query:
                    break
            # coreferences found for query
            if samples_per_query != 0:
                to_concat = pd.concat(
                    [q] * samples_per_query,
                    ignore_index=True,
                )
                queries = pd.concat([queries, to_concat], ignore_index=True)
            # break if dataset has desired size
            if total_dataset_size >= self.size:
                break

        # reset index to merge with queries
        passages = set_new_index(passages)
        queries = set_new_index(queries)

        # concat passages and queries dataframes
        coref_res_df = pd.concat([passages, queries], sort=False, axis=1)

        dataset_dict = encode_coref_res_dataset_to_json(
            coref_res_df, targets, SRC_DATASETS[self.source]["long"]
        )

        self.write_dataset_to_file("coref_res", dataset_dict)

    def fact_checking_dataset_creation(self) -> None:
        assert isinstance(self.split, str), "Split has to be defined for fever dataset"
        corpus_dfs, queries_dfs, qrels = sample_fever_data(self.split, self.size)
        json_res = encode_fact_checking_dataset_to_json(corpus_dfs, queries_dfs, qrels)

        for set_name, dataset in json_res.items():
            self.write_dataset_to_file("fact_checking", dataset, set_name)


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
