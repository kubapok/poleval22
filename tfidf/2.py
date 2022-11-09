import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = 'DATA_PROCESSED'


with open(DATA_DIR + '/vectorizer_wiki.pkl','rb') as f_out:
    vectorizer_wiki = pickle.load(f_out)

with open(DATA_DIR + '/vectorizer_all.pkl','rb') as f_out:
    vectorizer_all = pickle.load(f_out)

with open(DATA_DIR + '/passages_text_transformed_wiki.pkl','rb') as f_out:
    passages_text_transformed_wiki = pickle.load(f_out)

with open(DATA_DIR + '/passages_text_transformed_legal.pkl','rb') as f_out:
    passages_text_transformed_legal = pickle.load(f_out)

with open(DATA_DIR + '/passages_text_transformed_allegro.pkl','rb') as f_out:
    passages_text_transformed_allegro = pickle.load(f_out)

with open(DATA_DIR + '/queries_text_transformed_dev.pkl','rb') as f_out:
    queries_text_transformed_dev = pickle.load(f_out)

with open(DATA_DIR + '/queries_text_transformed_test_wiki.pkl','rb') as f_out:
    queries_text_transformed_test_wiki = pickle.load(f_out)

with open(DATA_DIR + '/queries_text_transformed_test_legal.pkl','rb') as f_out:
    queries_text_transformed_test_legal = pickle.load(f_out)

with open(DATA_DIR + '/queries_text_transformed_test_allegro.pkl','rb') as f_out:
    queries_text_transformed_test_allegro = pickle.load(f_out)

with open(DATA_DIR + '/queries_text_transformed_train.pkl','rb') as f_out:
    queries_text_transformed_train = pickle.load(f_out)

with open(DATA_DIR + '/df_passages_wiki.pkl','rb') as f_out:
    df_passages_wiki = pickle.load(f_out)

with open(DATA_DIR + '/df_passages_legal.pkl','rb') as f_out:
    df_passages_legal = pickle.load(f_out)

with open(DATA_DIR + '/df_passages_allegro.pkl','rb') as f_out:
    df_passages_allegro = pickle.load(f_out)

NR_OF_INDICES=10
def run(queries_text_transformed, df_passages, passages_text_transformed, f_out_path):
    top10_indices = []
    bs_size = 500
    for bs_in in tqdm(range(0, queries_text_transformed.shape[0], bs_size)):
        queries_batch = queries_text_transformed[bs_in: bs_in +bs_size]
        sims = cosine_similarity(queries_batch, passages_text_transformed)
        top10_indices_batch = np.argsort(-sims)[:,:NR_OF_INDICES].tolist()
        top10_indices += top10_indices_batch

    f_out = open(f_out_path, 'w')
    for i in tqdm(range(len(top10_indices))):
        o = df_passages.iloc[top10_indices[i]]['id'].tolist()
        o = [str(a) for a in o]
        f_out.write('\t'.join(o) + '\n')
    f_out.close()

run(queries_text_transformed_train, df_passages_wiki, passages_text_transformed_wiki,  '../../data/2022-passage-retrieval-tfidf/train/out.tsv')
run(queries_text_transformed_dev, df_passages_wiki, passages_text_transformed_wiki, '../../data/2022-passage-retrieval-tfidf/dev-0/out.tsv')
run(queries_text_transformed_test_wiki, df_passages_wiki, passages_text_transformed_wiki,  '../../data/2022-passage-retrieval-tfidf/test-A-wiki/out.tsv')
run(queries_text_transformed_test_legal, df_passages_legal, passages_text_transformed_legal,  '../../data/2022-passage-retrieval-tfidf/test-A-legal/out.tsv')
run(queries_text_transformed_test_allegro, df_passages_allegro, passages_text_transformed_allegro,  '../../data/2022-passage-retrieval-tfidf/test-A-allegro/out.tsv')
