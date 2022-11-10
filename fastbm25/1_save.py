import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from stopwords import STOPWORDS
from fastbm25 import fastbm25
from tokenizer_function import tokenize
from stempel import StempelStemmer


DATA_PROCESSED = 'DATA_PROCESSED'

DATA_DIR = '../../data/poleval-passage-retrieval'

df_passages_wiki = pd.read_json(DATA_DIR + '/wiki-trivia/passages.jl', lines=True)
df_passages_wiki['passage_id'] = df_passages_wiki['meta'].apply(lambda x : x['passage_id'])
df_passages_wiki['article_id'] = df_passages_wiki['meta'].apply(lambda x : x['article_id'])
df_passages_wiki['set'] = df_passages_wiki['article_id'].astype(str) + '-' + df_passages_wiki['passage_id'].astype(str)

df_passages_legal = pd.read_json(DATA_DIR + '/legal-questions/passages.jl', lines=True)
df_passages_allegro = pd.read_json(DATA_DIR + '/allegro-faq/passages.jl', lines=True)

df_queries_train = pd.read_csv('../../data/2022-passage-retrieval/train/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_dev = pd.read_csv('../../data/2022-passage-retrieval/dev-0/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_test_wiki = pd.read_csv('../../data/2022-passage-retrieval/test-A-wiki/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_test_legal = pd.read_csv('../../data/2022-passage-retrieval/test-A-legal/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_test_allegro = pd.read_csv('../../data/2022-passage-retrieval/test-A-allegro/in.tsv', sep =  '\t', names = ('source', 'text'))
print('reading done')


corpora_wiki = list(df_passages_wiki['text'])
corpora_wiki = [tokenize(doc) for doc in corpora_wiki]

corpora_legal = list(df_passages_legal['text'])
corpora_legal = [tokenize(doc) for doc in corpora_legal]

corpora_allegro = list(df_passages_allegro['text'])
corpora_allegro = [tokenize(doc) for doc in corpora_allegro]

print('corpora processing done')


bm25_wiki = fastbm25(corpora_wiki)
bm25_legal = fastbm25(corpora_legal)
bm25_allegro = fastbm25(corpora_allegro)
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
