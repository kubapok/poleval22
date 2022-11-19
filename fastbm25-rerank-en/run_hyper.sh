PARAMS='stemmer=default,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=0,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=default,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=default,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=default,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=0,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=default,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=1,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

wait



PARAMS='stemmer=default,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=0,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=default,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=default,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 


PARAMS='stemmer=default,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=0,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=default,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

wait




PARAMS='stemmer=polimorf,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=0,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=polimorf,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 


PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=0,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=1,PARAM_K1=1.5,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

wait



PARAMS='stemmer=polimorf,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=0,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=polimorf,word=0,word_lower=0,stemmed=0,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 

PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=0,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 


PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=0,stemmed_lower=0,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
python dev_run.py $PARAMS & 
wait
