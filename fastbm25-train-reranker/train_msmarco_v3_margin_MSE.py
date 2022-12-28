'''
This example shows how to train a SOTA Bi-Encoder with Margin-MSE loss for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

In this example we use a knowledge distillation setup. Sebastian Hofstätter et al. trained in https://arxiv.org/abs/2010.02666 an
an ensemble of large Transformer models for the MS MARCO datasets and combines the scores from a BERT-base, BERT-large, and ALBERT-large model.

We use the MSMARCO Hard Negatives File (Provided by Nils Reimers): https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz
Negative passage are hard negative examples, that were mined using different dense embedding, cross-encoder methods and lexical search methods.
Contains upto 50 negatives for each of the four retrieval systems: [bm25, msmarco-distilbert-base-tas-b, msmarco-MiniLM-L-6-v3, msmarco-distilbert-base-v3]
Each positive and negative passage comes with a score from a Cross-Encoder (msmarco-MiniLM-L-6-v3). This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

This example has been taken from here with few modifications to train SBERT (MSMARCO-v3) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v3.py)

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using dot-product to find matching passages for a given query.

For training, we use Margin MSE Loss. There, we pass triplets in the format:
triplets: (query, positive_passage, negative_passage)
label: positive_ce_score - negative_ce_score => (ce-score b/w query and positive or negative_passage)

PS: Using Margin MSE Loss doesn't require a threshold, or to set maximum negatives per system (required for Multiple Ranking Negative Loss)!
This is often a cumbersome process to find the optimal threshold which is dependent for Multiple Negative Ranking Loss.

Running this script:
python train_msmarco_v3_margin_MSE.py
'''

from sentence_transformers import SentenceTransformer, models, InputExample
from beir import util, LoggingHandler, losses
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
import pathlib, os, gzip, json
import logging
import random
import pickle
from torch.utils.data import DataLoader

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download msmarco.zip dataset and unzip the dataset

#################################
#### Parameters for Training ####
#################################

train_batch_size = 75           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n))
max_seq_length = 350            # Max length for passages. Increasing it, requires more GPU memory (O(n^2))

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

with open('train_dataset_for_rerank.pickle','rb') as f_in:
    text_pair_data_train = pickle.load(f_in)

train_samples = []
for label, query, passage in text_pair_data_train:
    train_samples.append(InputExample(texts=[query, passage], label=int(label)))
#### Load the hard negative MSMARCO jsonl triplets from SBERT 
#### These contain a ce-score which denotes the cross-encoder score for the query and passage.



# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.


# We construct the SentenceTransformer bi-encoder from scratch with mean-pooling
model_name = "distilbert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=train_batch_size)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

#### Training SBERT with dot-product (default) using Margin MSE Loss
train_loss = losses.MarginMSELoss(model=retriever.model)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v3-margin-MSE-loss".format(model_name))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 11
evaluation_steps = 100
warmup_steps = 100

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)
