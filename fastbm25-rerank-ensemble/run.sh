PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
CHALLENGEDIR='../../data/2022-passage-retrieval-ensemble'
#PARAMS=$1

export CUDA_VISIBLE_DEVICES=1
#bash 0.sh
#python 1_save.py  $PARAMS



#python 21.py 1 $PARAMS  $CHALLENGEDIR 
#python 21.py 3 $PARAMS  $CHALLENGEDIR
#python 21.py 4 $PARAMS  $CHALLENGEDIR
#python 21.py 2 $PARAMS  $CHALLENGEDIR
#
#python 22.py 3 $PARAMS  $CHALLENGEDIR
#python 22.py 4 $PARAMS  $CHALLENGEDIR
#python 22.py 1 $PARAMS  $CHALLENGEDIR 
#python 22.py 2 $PARAMS  $CHALLENGEDIR
#
#python 23.py 3 $PARAMS  $CHALLENGEDIR
#python 23.py 4 $PARAMS  $CHALLENGEDIR
#python 23.py 1 $PARAMS  $CHALLENGEDIR 
#python 23.py 2 $PARAMS  $CHALLENGEDIR
#
#python 20.py 3 $PARAMS $CHALLENGEDIR
#python 20.py 4 $PARAMS $CHALLENGEDIR
#python 20.py 1 $PARAMS $CHALLENGEDIR
#python 20.py 2 $PARAMS $CHALLENGEDIR


python 3.py 1 $PARAMS $CHALLENGEDIR
python 3.py 2 $PARAMS $CHALLENGEDIR
python 3.py 3 $PARAMS $CHALLENGEDIR
python 3.py 4 $PARAMS $CHALLENGEDIR


wait
