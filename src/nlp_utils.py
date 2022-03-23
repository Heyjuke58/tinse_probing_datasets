from nltk.tokenize import word_tokenize
from typing import List, Dict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pathlib import Path
import logging
import numpy as np

PORTER_STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))

def preprocess(string: str) -> List[str]:
    """
    Tokenize string, remove stopwords, stemming

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

def preprocess_sem_sim(string: str) -> List[str]:
    """
    Tokenize string, remove stopwords, convert to lowercase
    Preprocessing for semantic similarity task
    """
    x = word_tokenize(string)
    x = [word for word in x if not word in STOPWORDS]
    x = list(map(str.lower, x))
    return x

def load_glove_model(glove_file: Path) -> Dict[str, np.ndarray]:
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