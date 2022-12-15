from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import pickle

embedder = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

DATA_DIR = '../../data/poleval-passage-retrieval-eng'
DATAOUTPUT = 'DATA_PROCESSED'


df_passages_wiki = pd.read_json(DATA_DIR + '/wiki-trivia/passages.jl-en', lines=True)
df_passages_wiki['passage_id'] = df_passages_wiki['meta'].apply(lambda x : x['passage_id'])
df_passages_wiki['article_id'] = df_passages_wiki['meta'].apply(lambda x : x['article_id'])
df_passages_wiki['set'] = df_passages_wiki['article_id'].astype(str) + '-' + df_passages_wiki['passage_id'].astype(str)

df_passages_legal = pd.read_json(DATA_DIR + '/legal-questions/passages.jl-en', lines=True)
df_passages_allegro = pd.read_json(DATA_DIR + '/allegro-faq/passages.jl-en', lines=True)

#df_queries_train = pd.read_csv('../../data/2022-passage-retrieval-eng/train/in.tsv-en', sep =  '\t', names = ('source', 'text'))
#df_queries_dev = pd.read_csv('../../data/2022-passage-retrieval-eng/dev-0/in.tsv-en', sep =  '\t', names = ('source', 'text'))
#df_queries_test_wiki = pd.read_csv('../../data/2022-passage-retrieval-eng/test-A-wiki/in.tsv-en', sep =  '\t', names = ('source', 'text'))
#df_queries_test_legal = pd.read_csv('../../data/2022-passage-retrieval-eng/test-A-legal/in.tsv-en', sep =  '\t', names = ('source', 'text'))
#df_queries_test_allegro = pd.read_csv('../../data/2022-passage-retrieval-eng/test-A-allegro/in.tsv-en', sep =  '\t', names = ('source', 'text'))
print('reading done')

list_wiki_passages = df_passages_wiki['text_en'].tolist()
list_allegro_passages = df_passages_allegro['text_en'].tolist()
list_legal_passages = df_passages_legal['text_en'].tolist()

corpus_embeddings_allegro = embedder.encode(list_allegro_passages, convert_to_tensor=True, device='cuda')
corpus_embeddings_legal = embedder.encode(list_legal_passages, convert_to_tensor=True,device='cuda')
print('wiki legal')
BATCH_SIZE=2048
corpus_embeddings_wiki = embedder.encode(list_wiki_passages, convert_to_tensor=True,batch_size=BATCH_SIZE,device='cuda',show_progress_bar=True)
print('wiki done')

with open(f'{DATAOUTPUT}/corpus_embeddings_allegro.pickle', 'wb') as f_out:
    pickle.dump(corpus_embeddings_allegro, f_out)

with open(f'{DATAOUTPUT}/corpus_embeddings_legal.pickle', 'wb') as f_out:
    pickle.dump(corpus_embeddings_legal, f_out)

with open(f'{DATAOUTPUT}/corpus_embeddings_wiki.pickle', 'wb') as f_out:
    pickle.dump(corpus_embeddings_wiki, f_out)

corpus = list_wiki_passages
corpus_embeddings = corpus_embeddings_wiki
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))
