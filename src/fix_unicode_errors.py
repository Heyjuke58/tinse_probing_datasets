import ftfy
import sys
from pathlib import Path

PATH_CORPUS = './assets/msmarco/passage_re_ranking/collection.tsv'
PATH_CLEANED_CORPUS = './assets/msmarco/passage_re_ranking/collection_cleaned.tsv'

with open(Path(PATH_CORPUS), 'r') as f:
    with open(Path(PATH_CLEANED_CORPUS), 'w') as w:
        for l in f:
            w.write(ftfy.fix_text(l))