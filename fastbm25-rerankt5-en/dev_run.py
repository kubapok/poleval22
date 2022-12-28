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
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizer_function import Tokenizer
from argument_parser import get_params_dict
import sys


PARAMS = get_params_dict(sys.argv[1])


DATA_PROCESSED = 'DATA_PROCESSED'

DATA_DIR = '../../data/poleval-passage-retrieval'

df_passages_wiki = pd.read_json(DATA_DIR + '/wiki-trivia/passages.jl', lines=True)
df_passages_wiki['passage_id'] = df_passages_wiki['meta'].apply(lambda x : x['passage_id'])
df_passages_wiki['article_id'] = df_passages_wiki['meta'].apply(lambda x : x['article_id'])
df_passages_wiki['set'] = df_passages_wiki['article_id'].astype(str) + '-' + df_passages_wiki['passage_id'].astype(str)


df_queries_train = pd.read_csv('../../data/2022-passage-retrieval/train/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_dev = pd.read_csv('../../data/2022-passage-retrieval/dev-0/in.tsv', sep =  '\t', names = ('source', 'text'))
print('reading done')

tokenizer = Tokenizer(PARAMS)

corpora_wiki = list(df_passages_wiki['text'])
corpora_wiki = [tokenizer.tokenize(doc) for doc in corpora_wiki]

print('corpora processing done')


bm25_wiki = fastbm25(corpora_wiki,PARAMS['PARAM_K1'],PARAMS['PARAM_B'],PARAMS['EPSILON'])
print('bm25 learnt')



DATA_DIR = 'DATA_PROCESSED'


tokenizer = Tokenizer(PARAMS)
NR_OF_INDICES=10
def run(df_passages, ranker, in_file, out_file, top_n):
    with open(out_file, 'w') as f_out, open(in_file) as f_in:
        top10_indices = []
        for line in tqdm(f_in):
            dataset, query = line.rstrip().split('\t')
            query = tokenizer.tokenize(query)
            scores = ranker.top_k_sentence(query,NR_OF_INDICES)
            top10_indices_batch = df_passages.iloc[[a[1] for a in scores], ]['id'].tolist()
            top10_indices.append(top10_indices_batch)

        for o in tqdm(top10_indices):
            o = [str(a) for a in o]
            f_out.write('\t'.join(o) + '\n')
            
run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-bm25/dev-0/in.tsv' , f'../../data/2022-passage-retrieval-fastbm25/dev-0/out-{sys.argv[1]}.tsv', NR_OF_INDICES)
