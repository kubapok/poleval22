from stempel import StempelStemmer
from nltk.tokenize import word_tokenize

stemmer = StempelStemmer.polimorf()
def tokenize(sentence):
    return [stemmer.stem(x).lower() for x in word_tokenize(sentence) if stemmer.stem(x)]


