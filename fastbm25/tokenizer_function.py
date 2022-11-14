from stempel import StempelStemmer
from nltk.tokenize import word_tokenize
from stopwords import STOPWORDS

#def tokenize(sentence):
#    return [stemmer.stem(x).lower() for x in word_tokenize(sentence) if stemmer.stem(x) and stemmer.stem(x) not in STOPWORDS]

class Tokenizer():
    def __init__(self,PARAMS):

        if PARAMS['stemmer']=='default':
            self.stemmer = StempelStemmer.default()
        elif PARAMS['stemmer']=='polimorf':
            self.stemmer = StempelStemmer.polimorf()

        self.PARAMS=PARAMS

    def tokenize(self,sentence):
        words = []
        for word in word_tokenize(sentence):
            word_lower = word.lower()
            stemmed = self.stemmer.stem(word)

            if stemmed:
                stemmed_lower = stemmed.lower()
            else:
                stemmed_lower = None

            if self.PARAMS['word']=='1':
                words.append(word)  
            if self.PARAMS['word_lower']=='1':
                words.append(word_lower)  
            if stemmed and self.PARAMS['stemmed']=='1':
                words.append(stemmed)  
            if stemmed_lower and self.PARAMS['stemmed_lower']=='1':
                words.append(stemmed_lower)  

        #words = list(set(words))
        if self.PARAMS['STOPWORDS']=='1':
            words=[x for x in words if x not in STOPWORDS]

        return words
