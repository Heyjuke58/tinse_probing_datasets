SRC_PRETRAINED_GLOVE = "./assets/glove/glove.6B.300d.txt"

SRC_MS_MARCO = {
    "short": "msmarco",
    "long": "msmarco passage re-ranking",
    "index_name": "msmarco3",
    # "path_corpus": "./assets/msmarco/passage_re_ranking/collection_sample_orig.tsv",
    "path_corpus": "./assets/msmarco/passage_re_ranking/collection.tsv",
    "path_queries": "./assets/msmarco/passage_re_ranking/queries.dev.tsv",
    "path_queries_train": "./assets/msmarco/passage_re_ranking/queries.train.tsv",
    # "path_queries": "./assets/msmarco/passage_re_ranking/queries.dev.small.tsv",
    "path_top1000": "./assets/msmarco/passage_re_ranking/top1000.dev",
    "ir_path": "msmarco-passage/train",
    "id_pairs": "./assets/msmarco/passage_re_ranking/msm_id_pairs.csv",
}

SRC_TREC = {}
SRC_FEVER = {
    "short": "fever",
    "long": "fever fact checking",
    "path_corpus": "./assets/fever/corpus.jsonl",
    # "path_corpus": "./assets/fever/corpus_sample.jsonl",
    "path_queries": "./assets/fever/queries.jsonl",
    "path_qrels_test": "./assets/fever/qrels/test.tsv",
    "path_qrels_val": "./assets/fever/qrels/dev.tsv",
    "path_qrels_train": "./assets/fever/qrels/train.tsv",
}

SRC_DATASETS = {"msmarco": SRC_MS_MARCO, "trec": SRC_TREC, "fever": SRC_FEVER}