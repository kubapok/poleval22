for k in 0.2 0.3 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 
do
	for b in 0.2 0.3 0.4
	do
			PARAMS=stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=$k,PARAM_B=$b,EPSILON=0.25
			sleep  60
			python dev_run.py $PARAMS  &
	done
	wait
	for b in 0.5 0.6 0.7
	do
			PARAMS=stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=$k,PARAM_B=$b,EPSILON=0.25
			sleep  60
			python dev_run.py $PARAMS  &
	done
	wait
	for b in 0.8 0.9 1.0
	do
			PARAMS=stemmer=polimorf,word=1,word_lower=1,stemmed=1,stemmed_lower=1,STOPWORDS=1,PARAM_K1=$k,PARAM_B=$b,EPSILON=0.25
			sleep  60
			python dev_run.py $PARAMS  &
	done
	wait
done
