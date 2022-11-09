bash 0.sh
python 1_save.py
python 2.py 1 &
python 2.py 2 &
python 2.py 3 &
python 2.py 4 &
wait
bash 3.sh
