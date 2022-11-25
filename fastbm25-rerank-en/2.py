import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizer_function import Tokenizer
from argument_parser import get_params_dict
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

DEVICE='cuda'
model_name='cross-encoder/mmarco-mdeberta-v3-base-5negs-v1'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer_transformer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
model.to(DEVICE)
model.eval()


DATA_DIR = 'DATA_PROCESSED'
PARAMS = get_params_dict(sys.argv[2])


tokenizer = Tokenizer(PARAMS)
NR_OF_INDICES=500


def run(df_passages, ranker, in_file, out_file, top_n):
    with open(out_file, 'w') as f_out, open(in_file) as f_in:
        top10_indices = []
        for line in tqdm(f_in):
            dataset, query_pl, query_en = line.rstrip().split('\t')
            query_pl_tokenized = tokenizer.tokenize(query_pl)
            scores = ranker.top_k_sentence(query_pl_tokenized,NR_OF_INDICES)
            top10_indices_batch = df_passages.iloc[[a[1] for a in scores], ]['id'].tolist()


            #reranking
            #text_en = df_passages.iloc[[a[1] for a in scores], ]['text_en'].tolist()
            #query_en = [query_en]*len(text_en)

            text_pl = df_passages.iloc[[a[1] for a in scores], ]['text'].tolist() # do wywalenia
            query_pl = [query_pl]*len(text_pl) # do wywalenia

            scores_transformer = list()

            bs=100
            for i in range(0,len(text_pl), bs):
                #features_transformer = tokenizer_transformer(query_en, text_en, padding=True, truncation=True, return_tensors='pt')
                features_transformer = tokenizer_transformer(query_pl[i:i+bs], text_pl[i:i+bs], padding=True, truncation='only_second',max_length=512, return_tensors='pt').to(DEVICE) # do wywalenia
                scores_transformer_batch = model(**features_transformer).logits
                scores_transformer_batch = (-scores_transformer_batch).squeeze().tolist()
                #scores_transformer_batch = [a[1] for a in scores_transformer_batch] # to tylko jak jest podwójny output w niektórych modelach!!!!
                scores_transformer += scores_transformer_batch[:]

            new_order = [top10_indices_batch[a] for a in np.argsort(scores_transformer)   ]

            # final score
            top10_indices.append(new_order)


        for o in tqdm(top10_indices):
            o = [str(a) for a in o]
            f_out.write('\t'.join(o) + '\n')
            
if sys.argv[1] == '1':

    with open(DATA_DIR + '/bm25_wiki.pkl','rb') as f_out:
        bm25_wiki = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-fastbm25-eng/dev-0/in.tsv-en' , f'../../data/2022-passage-retrieval-fastbm25-eng/dev-0/out-{sys.argv[2]}.tsv', NR_OF_INDICES)

elif sys.argv[1] == '2':

    with open(DATA_DIR + '/bm25_wiki.pkl','rb') as f_out:
        bm25_wiki = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-fastbm25-eng/test-A-wiki/in.tsv-en' , f'../../data/2022-passage-retrieval-fastbm25-eng/test-A-wiki/out-{sys.argv[2]}.tsv', NR_OF_INDICES)

elif sys.argv[1] == '3':

    with open(DATA_DIR + '/bm25_legal.pkl','rb') as f_out:
        bm25_legal = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_legal.pkl','rb') as f_out:
        df_passages_legal = pickle.load(f_out)

    run(df_passages_legal, bm25_legal,'../../data/2022-passage-retrieval-fastbm25-eng/test-A-legal/in.tsv-en' , f'../../data/2022-passage-retrieval-fastbm25-eng/test-A-legal/out-{sys.argv[2]}.tsv', NR_OF_INDICES)

elif sys.argv[1] == '4':

    with open(DATA_DIR + '/bm25_allegro.pkl','rb') as f_out:
        bm25_allegro = pickle.load(f_out)

    with open(DATA_DIR + '/df_passages_allegro.pkl','rb') as f_out:
        df_passages_allegro = pickle.load(f_out)

    run(df_passages_allegro, bm25_allegro,'../../data/2022-passage-retrieval-fastbm25-eng/test-A-allegro/in.tsv-en' , f'../../data/2022-passage-retrieval-fastbm25-eng/test-A-allegro/out-{sys.argv[2]}.tsv', NR_OF_INDICES)
