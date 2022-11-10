import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizer_function import tokenize

DATA_DIR = 'DATA_PROCESSED'



NR_OF_INDICES=10
def run(df_passages, ranker, in_file, out_file, top_n):
    with open(out_file, 'w') as f_out, open(in_file) as f_in:
        top10_indices = []
        for line in tqdm(f_in):
            dataset, query = line.rstrip().split('\t')
            #query = word_tokenize(query.lower())
            #query = [stemmer.stem(x).lower() for x in word_tokenize(query) if stemmer.stem(x)]
            query = tokenize(query)
            scores = ranker.top_k_sentence(query,NR_OF_INDICES)
            #import pdb; pdb.set_trace()
            #top10_indices_batch = np.argsort(-scores)[:NR_OF_INDICES].tolist()
            top10_indices_batch = df_passages.iloc[[a[1] for a in scores], ]['id'].tolist()
            top10_indices.append(top10_indices_batch)

        for o in tqdm(top10_indices):
            o = [str(a) for a in o]
            f_out.write('\t'.join(o) + '\n')
            
if sys.argv[1] == '1':

    with open(DATA_DIR + '/bm25_wiki.pkl','rb') as f_out:
        bm25_wiki = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-bm25/dev-0/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/dev-0/out.tsv', NR_OF_INDICES)

elif sys.argv[1] == '2':

    with open(DATA_DIR + '/bm25_wiki.pkl','rb') as f_out:
        bm25_wiki = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-bm25/test-A-wiki/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/test-A-wiki/out.tsv', NR_OF_INDICES)

elif sys.argv[1] == '3':

    with open(DATA_DIR + '/bm25_legal.pkl','rb') as f_out:
        bm25_legal = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_legal.pkl','rb') as f_out:
        df_passages_legal = pickle.load(f_out)

    run(df_passages_legal, bm25_legal,'../../data/2022-passage-retrieval-bm25/test-A-legal/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/test-A-legal/out.tsv', NR_OF_INDICES)

elif sys.argv[1] == '4':

    with open(DATA_DIR + '/bm25_allegro.pkl','rb') as f_out:
        bm25_allegro = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_allegro.pkl','rb') as f_out:
        df_passages_allegro = pickle.load(f_out)

    run(df_passages_allegro, bm25_allegro,'../../data/2022-passage-retrieval-bm25/test-A-allegro/in.tsv' , '../../data/2022-passage-retrieval-fastbm25/test-A-allegro/out.tsv', NR_OF_INDICES)
