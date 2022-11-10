from stempel import StempelStemmer
from nltk.tokenize import word_tokenize
from stopwords import STOPWORDS

stemmer = StempelStemmer.polimorf()
def tokenize(sentence):
    return [stemmer.stem(x).lower() for x in word_tokenize(sentence) if stemmer.stem(x) and stemmmer.stem(x) not in STOPWORDS]


