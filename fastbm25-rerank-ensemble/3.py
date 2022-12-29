
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizer_function import Tokenizer
from argument_parser import get_params_dict
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import re

DEVICE='cuda'



DATA_DIR = '../fastbm25-rerank-en/DATA_PROCESSED'
PARAMS = get_params_dict(sys.argv[2])
CHALLENGEDIR = sys.argv[3]
NR_OF_INDICES=500


def run(in_file, out_file, model1_ranks,model2_ranks,model3_ranks):
    with open(out_file, 'w') as f_out, open(in_file,'rb') as f_in:
        output_scores = []
        top10_indices = pickle.load(f_in)
        for top10_indices_batch,m1,m2,m3  in zip(top10_indices,model1_ranks, model2_ranks, model3_ranks):

            scores = [a3 for a1,a2,a3  in zip(m1,m2,m3)]
            new_order = [top10_indices_batch[a] for a in np.argsort(scores)   ]
            new_order = [str(a) for a in new_order[:10]]
            f_out.write('\t'.join(new_order) + '\n')


            
if sys.argv[1] == '1':

    with open(f'{CHALLENGEDIR}/dev-0/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
        model1_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/dev-0/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
        model2_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/dev-0/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
        model3_ranks = pickle.load(f_in)


    run(f'{CHALLENGEDIR}/dev-0/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/dev-0/out.tsv', model1_ranks, model2_ranks, model3_ranks)

elif sys.argv[1] == '2':
    with open(f'{CHALLENGEDIR}/test-A-wiki/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
        model1_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/test-A-wiki/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
        model2_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/test-A-wiki/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
        model3_ranks = pickle.load(f_in)


    run(f'{CHALLENGEDIR}/test-A-wiki/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/test-A-wiki/out.tsv', model1_ranks, model2_ranks, model3_ranks)

elif sys.argv[1] == '3':
    with open(f'{CHALLENGEDIR}/test-A-legal/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
        model1_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/test-A-legal/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
        model2_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/test-A-legal/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
        model3_ranks = pickle.load(f_in)


    run(f'{CHALLENGEDIR}/test-A-legal/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/test-A-legal/out.tsv', model1_ranks, model2_ranks, model3_ranks)

elif sys.argv[1] == '4':
    with open(f'{CHALLENGEDIR}/test-A-allegro/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
        model1_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/test-A-allegro/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
        model2_ranks = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/test-A-allegro/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
        model3_ranks = pickle.load(f_in)


    run(f'{CHALLENGEDIR}/test-A-allegro/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/test-A-allegro/out.tsv', model1_ranks, model2_ranks, model3_ranks)
