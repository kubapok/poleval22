PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
#PARAMS=$1

export CUDA_VISIBLE_DEVICES=3
#bash 0.sh
#python 1_save.py  $PARAMS
python 2.py 1 $PARAMS  
python 2.py 2 $PARAMS  
python 2.py 3 $PARAMS  
python 2.py 4 $PARAMS  

wait
bash 3.sh
