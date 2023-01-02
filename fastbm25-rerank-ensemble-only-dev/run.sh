PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
CHALLENGEDIR='../../data/2022-passage-retrieval-ensemble-only-dev'
#PARAMS=$1

export CUDA_VISIBLE_DEVICES=0
#bash 0.sh
#python 1_save.py  $PARAMS


#python 2.py 1 $PARAMS  $CHALLENGEDIR /mnt/gpu_data1/kubapok/cache/models--cross-encoder--mmarco-mdeberta-v3-base-5negs-v1/snapshots/e4639f2fcee3da997e7da0a0948229ac172f83b1
#python 2.py 2 $PARAMS  $CHALLENGEDIR /mnt/gpu_data1/kubapok/cache/models--cross-encoder--mmarco-mdeberta-v3-base-5negs-v1/snapshots/e4639f2fcee3da997e7da0a0948229ac172f83b1
#
#python 2.py 1 $PARAMS  $CHALLENGEDIR cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
#python 2.py 2 $PARAMS  $CHALLENGEDIR cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
#
#python 2.py 1 $PARAMS  $CHALLENGEDIR '/mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/cross-encoder-mmarco-mdeberta-v3-base-5negs-v1-2022-12-12_08-57-01'
#python 2.py 2 $PARAMS  $CHALLENGEDIR '/mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/cross-encoder-mmarco-mdeberta-v3-base-5negs-v1-2022-12-12_08-57-01'
#
#python 2.py 1 $PARAMS  $CHALLENGEDIR /mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/-mnt-gpu_data1-kubapok-cache-models--cross-encoder--mmarco-mdeberta-v3-base-5negs-v1-snapshots-e4639f2fcee3da997e7da0a0948229ac172f83b1-2022-12-28_08-04-00
#python 2.py 2 $PARAMS  $CHALLENGEDIR /mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/-mnt-gpu_data1-kubapok-cache-models--cross-encoder--mmarco-mdeberta-v3-base-5negs-v1-snapshots-e4639f2fcee3da997e7da0a0948229ac172f83b1-2022-12-28_08-04-00
#
#
python 2.py 1 $PARAMS  $CHALLENGEDIR /mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1-2022-12-30_08-52-34
python 2.py 2 $PARAMS  $CHALLENGEDIR /mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1-2022-12-30_08-52-34




#python 20.py 1 $PARAMS  $CHALLENGEDIR
#python 20.py 2 $PARAMS  $CHALLENGEDIR


#python 3.py 1 $PARAMS $CHALLENGEDIR
#python 3.py 2 $PARAMS $CHALLENGEDIR


wait
