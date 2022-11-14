DATA=/mnt/gpu_data1/kubapok/poleval2022/data/2022-passage-retrieval-fastbm25

cat $DATA/test-A-wiki/out*.tsv $DATA/test-A-legal/out*.tsv $DATA/test-A-allegro/out*.tsv > $DATA/test-A/out.tsv
