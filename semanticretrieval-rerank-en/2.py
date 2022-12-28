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
from sentence_transformers import SentenceTransformer, util

DATA_DIR_EMBEDDINGS='/mnt/gpu_data1/kubapok/poleval2022/solutions/biencoder/DATA_PROCESSED'
MODEL_EMBEDDER='all-mpnet-base-v2'
embedder = SentenceTransformer(MODEL_EMBEDDER)

DEVICE='cuda'


model_name1 = sys.argv[4]
model1 = AutoModelForSequenceClassification.from_pretrained(model_name1)
tokenizer_transformers1 = AutoTokenizer.from_pretrained(model_name1,use_fast=False)
model1.to(DEVICE)
model1.eval()

model_name2 = sys.argv[5]
model2 = AutoModelForSequenceClassification.from_pretrained(model_name2)
tokenizer_transformers2 = AutoTokenizer.from_pretrained(model_name2,use_fast=False)
model2.to(DEVICE)
model2.eval()


DATA_DIR_BM25 = 'DATA_PROCESSED'
PARAMS = get_params_dict(sys.argv[2])
CHALLENGEDIR = sys.argv[3]

tokenizer_okapi = Tokenizer(PARAMS)
NR_OF_INDICES=3000

def get_reranked_scores(model, tokenizer_transformer, query_pl, text_pl):
    bs=30
    scores_transformer = list()
    for i in range(0,len(text_pl), bs):
        features_transformer = tokenizer_transformer(query_pl[i:i+bs], text_pl[i:i+bs], padding=True, truncation='only_second',max_length=512, return_tensors='pt').to(DEVICE)
        scores_transformer_batch = model(**features_transformer).logits
        scores_transformer_batch = (-scores_transformer_batch).squeeze(1).tolist()
        try:
            scores_transformer += scores_transformer_batch[:]
        except:
            import pdb; pdb.set_trace()
    return scores_transformer

def bm25_retrieve(query_pl, ranker):
    query_pl_tokenized = tokenizer_okapi.tokenize(query_pl)
    scores = ranker.top_k_sentence(query_pl_tokenized, NR_OF_INDICES)
    ids = [a[1] for a in scores]
    return ids

def bienecoder_retrieve(embedder, corpus_embeddings, query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=NR_OF_INDICES)
    return top_results[1].cpu().tolist()

def run(df_passages, ranker, corpus_embeddings, in_file, out_file, top_n):
    min_model1=100
    max_model1=-100

    min_model2=100
    max_model2=-100
    with open(out_file, 'w') as f_out, open(in_file) as f_in:
        top10_indices = []
        for line in tqdm(f_in, total=599):
            dataset, query_pl, query_en = line.rstrip().split('\t')


            ids_from_bm25 = bm25_retrieve(query_pl, ranker)
            ids_from_ss = bienecoder_retrieve(embedder,corpus_embeddings,query_en)

            ids = list(set(ids_from_bm25+ids_from_ss))

            to_ids_df_naming = df_passages.iloc[ids, ]['id'].tolist()


            #reranking
            #text_en = df_passages.iloc[[a[1] for a in scores], ]['text_en'].tolist()
            #query_en = [query_en]*len(text_en)

            text_pl = df_passages.iloc[ids, ]['text'].tolist() # do wywalenia
            query_pl = [query_pl]*len(text_pl) # do wywalenia

            scores_transformer1 = get_reranked_scores(model1, tokenizer_transformers1, query_pl, text_pl)
            scores_transformer2 = get_reranked_scores(model2, tokenizer_transformers2, query_pl, text_pl)
            scores_transformer = [(s1+s2)/2 for s1,s2 in zip(scores_transformer1, scores_transformer2)]
            #scores_transformer = [(3*s1+2*s2)/6 for s1,s2 in zip(scores_transformer1, scores_transformer2)]
            #scores_transformer = get_reranked_scores(model1, tokenizer_transformers1, query_pl, text_pl)

            

            #if min(scores_transformer1) < min_model1 :
            #    min_model1 = min(scores_transformer1)

            #if max(scores_transformer1) > max_model1 :
            #    max_model1 = max(scores_transformer1)

            #if min(scores_transformer2) < min_model2 :
            #    min_model2 = min(scores_transformer2)

            #if max(scores_transformer2) > max_model2 :
            #    max_model2 = max(scores_transformer2)

            new_order = [to_ids_df_naming[a] for a in np.argsort(scores_transformer)   ]

            # final score
            top10_indices.append(new_order)


        for o in tqdm(top10_indices):
            o = [str(a) for a in o]
            f_out.write('\t'.join(o) + '\n')
    print('model1')
    print(min_model1)
    print(max_model1)

    print('model2')
    print(min_model2)
    print(max_model2)
            
if sys.argv[1] == '1':

    with open(DATA_DIR_BM25 + '/bm25_wiki.pkl', 'rb') as f_out:
        bm25_wiki = pickle.load(f_out)

    with open(DATA_DIR_BM25 + '/df_passages_wiki.pkl', 'rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    with open(f'{DATA_DIR_EMBEDDINGS}/corpus_embeddings_wiki_{MODEL_EMBEDDER}.pickle', 'rb') as f_out:
        corpus_embeddings_wiki = pickle.load(f_out)

    run(df_passages_wiki, bm25_wiki, corpus_embeddings_wiki, f'{CHALLENGEDIR}/dev-0/in.tsv-en' , f'{CHALLENGEDIR}/dev-0/out-{sys.argv[2]}.tsv', NR_OF_INDICES)

elif sys.argv[1] == '2':

    with open(DATA_DIR_BM25 + '/bm25_wiki.pkl', 'rb') as f_out:
        bm25_wiki = pickle.load(f_out)

    with open(DATA_DIR_BM25 + '/df_passages_wiki.pkl', 'rb') as f_out:
        df_passages_wiki = pickle.load(f_out)

    with open(f'{DATA_DIR_EMBEDDINGS}/corpus_embeddings_wiki_{MODEL_EMBEDDER}.pickle', 'rb') as f_out:
        corpus_embeddings_wiki = pickle.load(f_out)

    run(df_passages_wiki, bm25_wiki, corpus_embeddings_wiki, f'{CHALLENGEDIR}/test-A-wiki/in.tsv-en' , f'{CHALLENGEDIR}/test-A-wiki/out-{sys.argv[2]}.tsv', NR_OF_INDICES)

elif sys.argv[1] == '3':

    with open(DATA_DIR_BM25 + '/bm25_legal.pkl', 'rb') as f_out:
        bm25_legal = pickle.load(f_out)

    with open(DATA_DIR_BM25 + '/df_passages_legal.pkl', 'rb') as f_out:
        df_passages_legal = pickle.load(f_out)


    with open(f'{DATA_DIR_EMBEDDINGS}/corpus_embeddings_legal_{MODEL_EMBEDDER}.pickle', 'rb') as f_out:
        corpus_embeddings_legal = pickle.load(f_out)


    run(df_passages_legal, bm25_legal, corpus_embeddings_legal, f'{CHALLENGEDIR}/test-A-legal/in.tsv-en' , f'{CHALLENGEDIR}/test-A-legal/out-{sys.argv[2]}.tsv', NR_OF_INDICES)

elif sys.argv[1] == '4':

    with open(DATA_DIR_BM25 + '/bm25_allegro.pkl', 'rb') as f_out:
        bm25_allegro = pickle.load(f_out)

    with open(DATA_DIR_BM25 + '/df_passages_allegro.pkl', 'rb') as f_out:
        df_passages_allegro = pickle.load(f_out)

    with open(f'{DATA_DIR_EMBEDDINGS}/corpus_embeddings_allegro_{MODEL_EMBEDDER}.pickle', 'rb') as f_out:
        corpus_embeddings_allegro = pickle.load(f_out)

    run(df_passages_allegro, bm25_allegro, corpus_embeddings_allegro, f'{CHALLENGEDIR}/test-A-allegro/in.tsv-en' , f'{CHALLENGEDIR}/test-A-allegro/out-{sys.argv[2]}.tsv', NR_OF_INDICES)

