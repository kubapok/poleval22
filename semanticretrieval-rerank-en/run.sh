PARAMS='stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=1.2,PARAM_B=0.75,EPSILON=0.25'
#CHALLENGEDIR=/mnt/gpu_data1/kubapok/poleval2022/data/2022-passage-retrieval-biencoder-retrieve-eng
#PARAMS=$1

export CUDA_VISIBLE_DEVICES=3
#bash 0.sh
#python 1_save.py  $PARAMS
python 2.py 1 $PARAMS  /mnt/gpu_data1/kubapok/poleval2022/data/2022-passage-retrieval-biencoder-retrieve-eng-wiki cross-encoder/mmarco-mdeberta-v3-base-5negs-v1 '/mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/cross-encoder-mmarco-mdeberta-v3-base-5negs-v1-2022-12-12_08-57-01'
python 2.py 2 $PARAMS  /mnt/gpu_data1/kubapok/poleval2022/data/2022-passage-retrieval-biencoder-retrieve-eng-wiki cross-encoder/mmarco-mdeberta-v3-base-5negs-v1 '/mnt/gpu_data1/kubapok/poleval2022/solutions/fastbm25-train-reranker/output/cross-encoder-mmarco-mdeberta-v3-base-5negs-v1-2022-12-12_08-57-01'
python 2.py 3 $PARAMS  /mnt/gpu_data1/kubapok/poleval2022/solutions/semanticretrieval-rerank-en-legal-allegro cross-encoder/mmarco-mdeberta-v3-base-5negs-v1 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
python 2.py 4 $PARAMS  /mnt/gpu_data1/kubapok/poleval2022/solutions/semanticretrieval-rerank-en-legal-allegro cross-encoder/mmarco-mdeberta-v3-base-5negs-v1 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1


wait
#bash 3.sh
