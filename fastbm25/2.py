import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize

DATA_DIR = 'DATA_PROCESSED'

with open(DATA_DIR + '/bm25_wiki.pkl','rb') as f_out:
    bm25_wiki = pickle.load(f_out)

with open(DATA_DIR + '/bm25_legal.pkl','rb') as f_out:
    bm25_legal = pickle.load(f_out)

with open(DATA_DIR + '/bm25_allegro.pkl','rb') as f_out:
    bm25_allegro = pickle.load(f_out)
#with open(DATA_DIR + '/corpora_all.pkl','rb') as f_out:
#    corpora_all = pickle.load(f_out)
#
#with open(DATA_DIR + '/corpora_wiki.pkl','rb') as f_out:
#    corpora_wiki = pickle.load(f_out)

with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
    df_passages_wiki = pickle.load(f_out)

with open(DATA_DIR + '/df_passages_legal.pkl','rb') as f_out:
    df_passages_legal = pickle.load(f_out)

with open(DATA_DIR + '/df_passages_allegro.pkl','rb') as f_out:
    df_passages_allegro = pickle.load(f_out)

print('reading pickles done')


NR_OF_INDICES=10
def run(df_passages, ranker, in_file, out_file, top_n):
    with open(out_file, 'w') as f_out, open(in_file) as f_in:
        top10_indices = []
        for line in tqdm(f_in):
            dataset, query = line.rstrip().split('\t')
            query = word_tokenize(query.lower())
            scores = ranker.get_scores(query)
            top10_indices_batch = np.argsort(-scores)[:NR_OF_INDICES].tolist()
            top10_indices.append(top10_indices_batch)

        for i in tqdm(range(len(top10_indices))):
            o = df_passages.iloc[top10_indices[i]]['id'].tolist()
            o = [str(a) for a in o]
            f_out.write('\t'.join(o) + '\n')
            
if sys.argv[1] == '1':
    run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-bm25/dev-0/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/dev-0/out.tsv', NR_OF_INDICES)
elif sys.argv[1] == '2':
    run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-bm25/test-A-wiki/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/test-A-wiki/out.tsv', NR_OF_INDICES)
elif sys.argv[1] == '3':
    run(df_passages_legal, bm25_legal,'../../data/2022-passage-retrieval-bm25/test-A-legal/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/test-A-legal/out.tsv', NR_OF_INDICES)
elif sys.argv[1] == '4':
    run(df_passages_allegro, bm25_allegro,'../../data/2022-passage-retrieval-bm25/test-A-allegro/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/test-A-allegro/out.tsv', NR_OF_INDICES)
