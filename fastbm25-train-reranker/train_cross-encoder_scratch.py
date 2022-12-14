"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import pickle
import torch
import torch.nn


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


#First, we define the transformer model we want to fine-tune
#model_name = 'distilroberta-base'
#model_name = 'cross-encoder/mmarco-mdeberta-v3-base-5negs-v1'
#model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

#model_name='cross-encoder/mmarco-mdeberta-v3-base-5negs-v1'
#train_batch_size = 64
#model_name = 'allegro/herbert-base-cased'
#model_name='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
model_name='cross-encoder/mmarco-mdeberta-v3-base-5negs-v1'
#model_name='xlm-roberta-large'
#model_name='allegro/herbert-base-cased'
#model_name='cross-encoder/mmarco-mdeberta-v3-base-5negs-v1'
#model_name='google/mt5-large'
train_batch_size = 64

num_epochs = 50
model_save_path = 'output/'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#We set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=512)

# TRAIN DATASET
with open('train_dataset_for_rerank_50_negs_1000.pickle','rb') as f_in:
#with open('train_dataset_for_rerank_100_negs_2000.pickle','rb') as f_in:
    text_pair_data_train = pickle.load(f_in)

train_samples = []
for label, query, passage in text_pair_data_train:
    train_samples.append(InputExample(texts=[query, passage], label=int(label)))

# DEV DATASET
with open('dev-0_dataset_for_rerank_5_negs_30.pickle','rb') as f_in:
    text_pair_data_dev = pickle.load(f_in)


dev_samples = dict()
for label, query, passage in text_pair_data_dev:
    if not query in dev_samples.keys():
        dev_samples[query] = dict()
        dev_samples[query]['query'] = query
        dev_samples[query]['positive'] = list()
        dev_samples[query]['negative'] = list()

    if int(label) == 0:
        dev_samples[query]['negative'].append(passage)
    elif int(label) == 1:
        dev_samples[query]['positive'].append(passage)
    else:
        assert False

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name='train-eval',mrr_at_k=30)

# Configure the training
warmup_steps = 1000
logging.info("Warmup-steps: {}".format(warmup_steps))

print(" TRENING SIE ZACZYNA")
# Train the model
model.fit(train_dataloader=train_dataloader,
          #loss_fct=torch.nn.MSELoss(),
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr':2e-6},
          scheduler = 'constantlr',
          use_amp=False)

# Save latest model
model.save(model_save_path+'-latest')
