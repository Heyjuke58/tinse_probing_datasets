from nltk.tokenize import word_tokenize
from typing import List
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

PORTER_STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))

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