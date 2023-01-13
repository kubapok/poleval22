
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
NR_OF_INDICES=1500


def run(in_file, out_file, models_dict):
    with open(out_file, 'w') as f_out, open(in_file,'rb') as f_in:
        output_scores = []
        top10_indices = pickle.load(f_in)
        for top10_indices_batch,m1,m2,m3,m4,m5  in zip(top10_indices,models_dict['mt5_base'],models_dict['deberta'],models_dict['mmini'],models_dict['mt53B'],models_dict['mt513B']   ):

            scores = [a4 for a1,a2,a3,a4,a5 in zip(m1,m2,m3,m4,m5)]
            new_order = [top10_indices_batch[a] for a in np.argsort(scores)   ]
            new_order = [str(a) for a in new_order[:10]]
            f_out.write('\t'.join(new_order) + '\n')

def load_models_dict(dataset_type):
    with open(f'{CHALLENGEDIR}/{dataset_type}/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
        mt5_base = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/{dataset_type}/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
        deberta = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/{dataset_type}/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
        mmini = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/{dataset_type}/out-mt5-3B-mmarco-en-pt.pickle','rb') as f_in:
        mt53B = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/{dataset_type}/out-mt5-13b-mmarco-100k.pickle','rb') as f_in:
        mt513B = pickle.load(f_in)
    with open(f'{CHALLENGEDIR}/{dataset_type}/out-mt5-13b-mmarco-100k.pickle','rb') as f_in:
        mt513B = pickle.load(f_in)
    #with open(f'{CHALLENGEDIR}/{dataset_type}/out-mt5-13b-mmarco-100k-kdd-alltrain-4e.pickle','rb') as f_in:
        #mt513B4e = pickle.load(f_in)
    #with open(f'{CHALLENGEDIR}/{dataset_type}/out-mt5-13b-mmarco-100k-kdd-alltrain-4.5e.pickle','rb') as f_in:
        #mt513B45e = pickle.load(f_in)
    d = {'mt5_base':mt5_base,
            'deberta':deberta,
            'mmini':mmini,
            'mt53B':mt53B,
            'mt513B':mt513B}
            #'mt513B4e':mt513B4e,
            #'mt513B45e':mt513B45e}
    return d

if sys.argv[1] == '1':
    dataset='dev-0'
elif sys.argv[1] == '2':
    dataset='test-A-wiki'
elif sys.argv[1] == '3':
    dataset='test-A-legal'
elif sys.argv[1] == '4':
    dataset='test-A-allegro'
else:
    assert False

models_dict = load_models_dict(dataset)
run(f'{CHALLENGEDIR}/{dataset}/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/{dataset}/out.tsv', models_dict)

#elif sys.argv[1] == '2':
#    with open(f'{CHALLENGEDIR}/test-A-wiki/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
#        model1_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-wiki/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
#        model2_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-wiki/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
#        model3_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-wiki/out-mt5-3b-mmarco-100k-kdd-alltrain-4.5e.pickle','rb') as f_in:
#        model4_ranks = pickle.load(f_in)
#
#
#    run(f'{CHALLENGEDIR}/test-A-wiki/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/test-A-wiki/out.tsv', model1_ranks, model2_ranks, model3_ranks, model4_ranks)
#
#elif sys.argv[1] == '3':
#    with open(f'{CHALLENGEDIR}/test-A-legal/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
#        model1_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-legal/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
#        model2_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-legal/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
#        model3_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-legal/out-mt5-3b-mmarco-100k-kdd-alltrain-4.5e.pickle','rb') as f_in:
#        model4_ranks = pickle.load(f_in)
#
#
#    run(f'{CHALLENGEDIR}/test-A-legal/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/test-A-legal/out.tsv', model1_ranks, model2_ranks, model3_ranks, model4_ranks)
#
#elif sys.argv[1] == '4':
#    with open(f'{CHALLENGEDIR}/test-A-allegro/out-mt5-base-mmarco-v2.pickle','rb') as f_in:
#        model1_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-allegro/out-e4639f2fcee3da997e7da0a0948229ac172f83b1.pickle','rb') as f_in:
#        model2_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-allegro/out-mmarco-mMiniLMv2-L12-H384-v1.pickle','rb') as f_in:
#        model3_ranks = pickle.load(f_in)
#    with open(f'{CHALLENGEDIR}/test-A-allegro/out-mt5-3b-mmarco-100k-kdd-alltrain-4.5e.pickle','rb') as f_in:
#        model4_ranks = pickle.load(f_in)
#
#
#    run(f'{CHALLENGEDIR}/test-A-allegro/rerank-indices-{NR_OF_INDICES}.pickle' , f'{CHALLENGEDIR}/test-A-allegro/out.tsv', model1_ranks, model2_ranks, model3_ranks, model4_ranks)
