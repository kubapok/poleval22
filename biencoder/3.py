import pickle
from sentence_transformers import SentenceTransformer, util
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

DATA_DIR = 'DATA_PROCESSED'
CHALLANGEDIR = sys.argv[2]




DEVICE='cuda'
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)


NR_OF_INDICES=10

def retrieve(embedder,  corpus_embeddings, query):
    
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=NR_OF_INDICES)
    return top_results[1].cpu().tolist()



def run(df_passages, corpus_embeddings, in_file, out_file, top_n):
    with open(out_file, 'w') as f_out, open(in_file) as f_in:
        top10_indices = []
        for line in tqdm(f_in):
            dataset, query_pl, query_en = line.rstrip().split('\t')

            results_indices = retrieve(embedder, corpus_embeddings, query_en)
            
            top10_indices = df_passages.iloc[results_indices]['id'].tolist()


        for o in tqdm(top10_indices):
            o = [str(a) for a in o]
            f_out.write('\t'.join(o) + '\n')

            
if sys.argv[1] == '1':

    with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    with open(f'{DATA_DIR}/corpus_embeddings_wiki.pickle', 'rb') as f_out:
        corpus_embeddings_wiki = pickle.load(f_out)

    run(df_passages_wiki, corpus_embeddings_wiki, f'{CHALLANGEDIR}/dev-0/in.tsv-en' , f'{CHALLANGEDIR}/dev-0/out.tsv', NR_OF_INDICES)

elif sys.argv[1] == '2':

    with open(f'{DATA_DIR}/corpus_embeddings_wiki.pickle', 'rb') as f_out:
        corpus_embeddings_wiki = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    run(df_passages_wiki, corpus_embeddings_wiki, f'{CHALLANGEDIR}/test-A-wiki/in.tsv-en' , f'{CHALLANGEDIR}/test-A-wiki/out.tsv', NR_OF_INDICES)

elif sys.argv[1] == '3':
    with open(f'{DATA_DIR}/corpus_embeddings_legal.pickle', 'rb') as f_out:
        corpus_embeddings_legal = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_legal.pkl','rb') as f_out:
        df_passages_legal = pickle.load(f_out)


    run(df_passages_legal, corpus_embeddings_legal, f'{CHALLANGEDIR}/test-A-legal/in.tsv-en' , f'{CHALLANGEDIR}/test-A-legal/out.tsv', NR_OF_INDICES)

elif sys.argv[1] == '4':
    with open(f'{DATA_DIR}/corpus_embeddings_allegro.pickle', 'rb') as f_out:
        corpus_embeddings_allegro = pickle.load(f_out)


    with open(DATA_DIR + '/df_passages_allegro.pkl','rb') as f_out:
        df_passages_allegro = pickle.load(f_out)

    run(df_passages_allegro, corpus_embeddings_allegro,  f'{CHALLANGEDIR}/test-A-allegro/in.tsv-en' , f'{CHALLANGEDIR}/test-A-allegro/out.tsv', NR_OF_INDICES)

