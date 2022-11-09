import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stopwords import STOPWORDS

DATA_PROCESSED = 'DATA_PROCESSED'

DATA_DIR = '../../data/poleval-passage-retrieval'

df_passages_wiki = pd.read_json(DATA_DIR + '/wiki-trivia/passages.jl', lines=True)
df_passages_wiki['passage_id'] = df_passages_wiki['meta'].apply(lambda x : x['passage_id'])
df_passages_wiki['article_id'] = df_passages_wiki['meta'].apply(lambda x : x['article_id'])
df_passages_wiki['set'] = df_passages_wiki['article_id'].astype(str) + '-' + df_passages_wiki['passage_id'].astype(str)

df_passages_legal = pd.read_json(DATA_DIR + '/legal-questions/passages.jl', lines=True)
df_passages_allegro = pd.read_json(DATA_DIR + '/allegro-faq/passages.jl', lines=True)

df_queries_train = pd.read_csv('../../data/2022-passage-retrieval/train/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_dev = pd.read_csv('../../data/2022-passage-retrieval/dev-0/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_test_wiki = pd.read_csv('../../data/2022-passage-retrieval/test-A-wiki/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_test_legal = pd.read_csv('../../data/2022-passage-retrieval/test-A-legal/in.tsv', sep =  '\t', names = ('source', 'text'))
df_queries_test_allegro = pd.read_csv('../../data/2022-passage-retrieval/test-A-allegro/in.tsv', sep =  '\t', names = ('source', 'text'))


#vectorizer_wiki = TfidfVectorizer(stop_words=STOPWORDS) 
vectorizer_wiki = TfidfVectorizer() 
vectorizer_wiki.fit( list(df_passages_wiki['text']) + list(df_queries_train['text']) )

vectorizer_legal = TfidfVectorizer() 
vectorizer_legal.fit( list(df_passages_legal['text']) + list(df_queries_train['text']) )


vectorizer_allegro = TfidfVectorizer() 
vectorizer_allegro.fit(list(df_passages_allegro['text']) + list(df_queries_train['text']) )

#vectorizer_all = TfidfVectorizer(stop_words=STOPWORDS) 
vectorizer_all = TfidfVectorizer() 
corpora_all=  (   list(df_passages_wiki['text'])
                + list(df_passages_legal['text'])
                + list(df_passages_allegro['text'])

                + list(df_queries_train['text'])  
                + list(df_queries_dev['text'])  
                + list(df_queries_test_wiki['text'])  
                + list(df_queries_test_allegro['text'])  
                + list(df_queries_test_legal['text']) )

vectorizer_all.fit(corpora_all)




passages_text_transformed_wiki = vectorizer_wiki.transform(df_passages_wiki['text'])
passages_text_transformed_legal = vectorizer_all.transform(df_passages_legal['text'])
passages_text_transformed_allegro = vectorizer_all.transform(df_passages_allegro['text'])



queries_text_transformed_train = vectorizer_wiki.transform(df_queries_train['text'])
queries_text_transformed_dev = vectorizer_wiki.transform(df_queries_dev['text'])
queries_text_transformed_test_wiki = vectorizer_wiki.transform(df_queries_test_wiki['text'])
queries_text_transformed_test_legal = vectorizer_all.transform(df_queries_test_legal['text'])
queries_text_transformed_test_allegro = vectorizer_all.transform(df_queries_test_allegro['text'])

#sims = cosine_similarity(queries_text_transformed, passages_text_transformed_wiki)

with open(f'{DATA_PROCESSED}/vectorizer_wiki.pkl','wb') as f_out:
    pickle.dump(vectorizer_wiki,f_out)

with open(f'{DATA_PROCESSED}/vectorizer_all.pkl','wb') as f_out:
    pickle.dump(vectorizer_all,f_out)

with open(f'{DATA_PROCESSED}/passages_text_transformed_wiki.pkl','wb') as f_out:
    pickle.dump(passages_text_transformed_wiki,f_out)

with open(f'{DATA_PROCESSED}/passages_text_transformed_legal.pkl','wb') as f_out:
    pickle.dump(passages_text_transformed_legal,f_out)

with open(f'{DATA_PROCESSED}/passages_text_transformed_allegro.pkl','wb') as f_out:
    pickle.dump(passages_text_transformed_allegro,f_out)

with open(f'{DATA_PROCESSED}/queries_text_transformed_train.pkl','wb') as f_out:
    pickle.dump(queries_text_transformed_train,f_out)

with open(f'{DATA_PROCESSED}/queries_text_transformed_dev.pkl','wb') as f_out:
    pickle.dump(queries_text_transformed_dev,f_out)

with open(f'{DATA_PROCESSED}/queries_text_transformed_test_wiki.pkl','wb') as f_out:
    pickle.dump(queries_text_transformed_test_wiki,f_out)

with open(f'{DATA_PROCESSED}/queries_text_transformed_test_legal.pkl','wb') as f_out:
    pickle.dump(queries_text_transformed_test_legal,f_out)

with open(f'{DATA_PROCESSED}/queries_text_transformed_test_allegro.pkl','wb') as f_out:
    pickle.dump(queries_text_transformed_test_allegro,f_out)

with open(f'{DATA_PROCESSED}/df_passages_wiki.pkl','wb') as f_out:
    pickle.dump(df_passages_wiki,f_out)

with open(f'{DATA_PROCESSED}/df_passages_legal.pkl','wb') as f_out:
    pickle.dump(df_passages_legal,f_out)

with open(f'{DATA_PROCESSED}/df_passages_allegro.pkl','wb') as f_out:
    pickle.dump(df_passages_allegro,f_out)
