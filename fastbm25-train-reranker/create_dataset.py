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
import random


DATA_DIR = 'DATA_PROCESSED'
PARAMS = get_params_dict(sys.argv[1])


tokenizer_okapi = Tokenizer(PARAMS)
NR_OF_INDICES=500
neg_pos_ratio=50


def run(df_passages, ranker, in_file, out_file, exp_file, top_n):
    with open(out_file, 'wb') as f_out, open(in_file) as f_in, open(exp_file) as f_exp:
        cnt = 0
        top10_indices = []
        train_dataset_for_rerank = []
        for line_in, line_exp in zip(f_in, f_exp):
            print(cnt)
            cnt +=1
            dataset, query_pl, query_en = line_in.rstrip().split('\t')
            query_pl_tokenized = tokenizer_okapi.tokenize(query_pl)
            scores = ranker.top_k_sentence(query_pl_tokenized,NR_OF_INDICES)
            top10_indices_batch = df_passages.iloc[[a[1] for a in scores], ]['id'].tolist()

            positives_text  = []
            positives_ids = line_exp.rstrip().split('\t')
            positives_text = df_passages_wiki[df_passages_wiki['id'].apply(lambda x: x in positives_ids)]['text'].tolist()

           
            negatives_candidates_ids = [a for a in top10_indices_batch if a not in positives_ids]
            # hard- no shuffle
            #random.shuffle(negatives_candidates_ids)
            negatives_candidates_ids = negatives_candidates_ids[:neg_pos_ratio*len(positives_text)]
            negatives_text  = df_passages_wiki[df_passages_wiki['id'].apply(lambda x: x in negatives_candidates_ids)]['text'].tolist()

            for text in positives_text:
                train_dataset_for_rerank.append([1,query_pl, text]) 

            for text in negatives_text:
                train_dataset_for_rerank.append([0,query_pl, text]) 

        random.shuffle(train_dataset_for_rerank) 
        pickle.dump(train_dataset_for_rerank, f_out)

with open(DATA_DIR + '/bm25_wiki.pkl','rb') as f_out:
    bm25_wiki = pickle.load(f_out)

with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
    df_passages_wiki = pickle.load(f_out)


run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-fastbm25-eng/dev-0/in.tsv-en' , f'dev-0_dataset_for_rerank_50_negs_500_hard.pickle', f'../../data//2022-passage-retrieval-fastbm25-eng/dev-0/expected.tsv', NR_OF_INDICES)
run(df_passages_wiki, bm25_wiki,'../../data/2022-passage-retrieval-fastbm25-eng/train/in.tsv-en' , f'train_dataset_for_rerank_50_negs_500_hard.pickle', f'../../data//2022-passage-retrieval-fastbm25-eng/train/expected.tsv', NR_OF_INDICES)
