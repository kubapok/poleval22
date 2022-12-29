import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from stopwords import STOPWORDS
from my_fastbm25 import fastbm25
from tokenizer_function import Tokenizer
from stempel import StempelStemmer
from argument_parser import get_params_dict
import sys


PARAMS = get_params_dict(sys.argv[1])


DATA_PROCESSED = 'DATA_PROCESSED'

#DATA_DIR = '../../data/poleval-passage-retrieval-eng-sample/'
DATA_DIR = '../../data/poleval-passage-retrieval-eng/'

df_passages_wiki = pd.read_json(DATA_DIR + '/wiki-trivia/passages.jl-en', lines=True)
df_passages_wiki['passage_id'] = df_passages_wiki['meta'].apply(lambda x : x['passage_id'])
df_passages_wiki['article_id'] = df_passages_wiki['meta'].apply(lambda x : x['article_id'])
df_passages_wiki['set'] = df_passages_wiki['article_id'].astype(str) + '-' + df_passages_wiki['passage_id'].astype(str)

df_passages_legal = pd.read_json(DATA_DIR + '/legal-questions/passages.jl-en', lines=True)
df_passages_allegro = pd.read_json(DATA_DIR + '/allegro-faq/passages.jl-en', lines=True)

#df_queries_train = pd.read_csv('../../data/2022-passage-retrieval/train/in.tsv', sep =  '\t', names = ('source', 'text'))
#df_queries_dev = pd.read_csv('../../data/2022-passage-retrieval/dev-0/in.tsv', sep =  '\t', names = ('source', 'text'))
#df_queries_test_wiki = pd.read_csv('../../data/2022-passage-retrieval/test-A-wiki/in.tsv', sep =  '\t', names = ('source', 'text'))
#df_queries_test_legal = pd.read_csv('../../data/2022-passage-retrieval/test-A-legal/in.tsv', sep =  '\t', names = ('source', 'text'))
#df_queries_test_allegro = pd.read_csv('../../data/2022-passage-retrieval/test-A-allegro/in.tsv', sep =  '\t', names = ('source', 'text'))
print('reading done')

tokenizer = Tokenizer(PARAMS)

corpora_wiki = list(df_passages_wiki['text'])
corpora_wiki = [tokenizer.tokenize(doc) for doc in corpora_wiki]

corpora_legal = list(df_passages_legal['text'])
corpora_legal = [tokenizer.tokenize(doc) for doc in corpora_legal]

corpora_allegro = list(df_passages_allegro['text'])
corpora_allegro = [tokenizer.tokenize(doc) for doc in corpora_allegro]

print('corpora processing done')


bm25_wiki = fastbm25(corpora_wiki,PARAMS['PARAM_K1'],PARAMS['PARAM_B'],PARAMS['EPSILON'])
bm25_legal = fastbm25(corpora_legal,PARAMS['PARAM_K1'],PARAMS['PARAM_B'],PARAMS['EPSILON'])
bm25_allegro = fastbm25(corpora_allegro,PARAMS['PARAM_K1'],PARAMS['PARAM_B'],PARAMS['EPSILON'])
print('bm25 learnt')


with open(f'{DATA_PROCESSED}/bm25_wiki.pkl','wb') as f_out:
    pickle.dump(bm25_wiki,f_out)

with open(f'{DATA_PROCESSED}/bm25_legal.pkl','wb') as f_out:
    pickle.dump(bm25_legal,f_out)

with open(f'{DATA_PROCESSED}/bm25_allegro.pkl','wb') as f_out:
    pickle.dump(bm25_allegro,f_out)

with open(f'{DATA_PROCESSED}/corpora_wiki.pkl','wb') as f_out:
    pickle.dump(corpora_wiki,f_out)

with open(f'{DATA_PROCESSED}/df_passages_wiki.pkl','wb') as f_out:
    pickle.dump(df_passages_wiki,f_out)

with open(f'{DATA_PROCESSED}/df_passages_legal.pkl','wb') as f_out:
    pickle.dump(df_passages_legal,f_out)

with open(f'{DATA_PROCESSED}/df_passages_allegro.pkl','wb') as f_out:
    pickle.dump(df_passages_allegro,f_out)
